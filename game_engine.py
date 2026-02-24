"""
game_engine.py
==============
Core game engine for AI Murder Mystery: The Blackwood Mansion.

Contains:
  MurderMysteryGame — the single orchestrating class that wires together
                      all agents, manages state, and exposes a clean API
                      consumed by both the Streamlit UI (app.py) and the
                      CLI runner (cli.py).

Public API summary:
    game = MurderMysteryGame()
    game.get_current_suspect()           → SuspectProfile
    game.switch_suspect(sid)             → bool
    game.interrogate(question)           → str
    game.plan_next_question(sid, topics) → str
    game.deduce_accusation()             → (sid, weapon, motive) | (None, None, None)
    game.make_accusation(sid, w, m)      → (won, score, eval_text)
    game.reset()                         → None

Logging
-------
Every significant event is emitted through the standard ``logging`` module
so that the host application (Streamlit, CLI, or any test harness) can route,
filter, and aggregate log output without changing this file.

Configure log level and destination once at your entry point, e.g.:

    # app.py / cli.py
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

The logger name for this module is ``murder_mystery.game_engine``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from agents import (
    build_accusation_agent,
    build_critique_agent,
    build_deduction_agent,
    build_interrogation_planner,
    build_suspect_agent,
)
from case_data import CASE_FILE, SUSPECTS, VALID_MOTIVES, VALID_WEAPONS
from config import CRITIQUE_TRIGGERS, GAME_CONFIG
from models import DeductionResult, GameState, SuspectProfile
from scoring import calculate_score

# ---------------------------------------------------------------------------
# Module-level logger
#
# Using a hierarchical name ("murder_mystery.game_engine") means callers can
# configure the root "murder_mystery" logger to capture all modules at once,
# or silence just this one individually.
# ---------------------------------------------------------------------------
logger = logging.getLogger("murder_mystery.game_engine")


class MurderMysteryGame:
    """
    Main game engine.

    Owns all five agent types and the shared GameState / conversation logs.
    The Streamlit UI interacts with this class exclusively; it has no direct
    awareness of Agno agents or Groq model calls.

    Attributes:
        state:                 Current GameState (turn counts, win flag, score).
        conversation_logs:     Dict mapping suspect_id to list of (question, answer) tuples.
        current_suspect_id:    The suspect currently selected for interrogation.
        suspect_agents:        Dict mapping suspect_id to their SuspectAgent instance.
        critique_agent:        The shared CritiqueAgent (one per game session).
        interrogation_planner: The InterrogationPlanner used in AI-agent mode.
        deduction_agent:       The DeductionAgent that analyses full transcripts.
        accusation_agent:      The AccusationAgent judge.
    """

    def __init__(self) -> None:
        self.state = GameState(max_turns=GAME_CONFIG.max_turns)

        # conversation_logs[sid] = [(question, answer), ...]
        self.conversation_logs: Dict[str, List[Tuple[str, str]]] = {
            sid: [] for sid in SUSPECTS
        }
        self.current_suspect_id: str = "s1"

        logger.info(
            "MurderMysteryGame initialised — case_id=%s, max_turns=%d",
            CASE_FILE.get("case_id", "unknown"),
            GAME_CONFIG.max_turns,
        )

        # Instantiate all agents once per game session.
        # Agents hold their own internal memory, so rebuilding them on reset
        # clears any prior context that could leak between sessions.
        self._build_all_agents()

    # ------------------------------------------------------------------
    # Internal agent construction
    # ------------------------------------------------------------------

    def _build_all_agents(self) -> None:
        """Instantiate (or re-instantiate) every agent for a clean session."""
        logger.debug("Building all agents for new session.")
        self.suspect_agents: Dict = {
            sid: build_suspect_agent(profile)
            for sid, profile in SUSPECTS.items()
        }
        self.critique_agent        = build_critique_agent()
        self.interrogation_planner = build_interrogation_planner()
        self.deduction_agent       = build_deduction_agent()
        self.accusation_agent      = build_accusation_agent()
        logger.debug(
            "All agents built: suspects=%s", list(self.suspect_agents.keys())
        )

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def get_current_suspect(self) -> SuspectProfile:
        """Return the SuspectProfile for the currently selected suspect."""
        return SUSPECTS[self.current_suspect_id]

    def switch_suspect(self, suspect_id: str) -> bool:
        """
        Switch the active interrogation target.

        Args:
            suspect_id: Must be a key in SUSPECTS (e.g. "s1").

        Returns:
            True if the switch succeeded, False if the ID was unrecognised.
        """
        if suspect_id not in SUSPECTS:
            logger.warning(
                "switch_suspect called with unknown suspect_id=%r. Valid IDs: %s",
                suspect_id,
                list(SUSPECTS.keys()),
            )
            return False
        logger.debug(
            "Switched suspect: %s -> %s", self.current_suspect_id, suspect_id
        )
        self.current_suspect_id = suspect_id
        return True

    # ------------------------------------------------------------------
    # Core interrogation loop
    # ------------------------------------------------------------------

    def interrogate(self, player_message: str) -> str:
        """
        Process a player question and return the suspect's (filtered) response.

        Interrogation flow:
          1. Build a context-aware history prefix for the suspect's prompt.
             After GAME_CONFIG.context_compress_after turns, older exchanges
             are summarised into bullet points to bound context window usage.
          2. Run the SuspectAgent to get a raw candidate response.
          3. Conditionally run the CritiqueAgent ONLY if the raw response
             contains a CRITIQUE_TRIGGERS keyword. Clean responses pass through
             without the extra model call, saving ~300 ms and token costs.
          4. Log the (question, filtered_answer) pair and advance the turn counter.

        Args:
            player_message: The question typed by the player (or generated
                            by the AI planner).

        Returns:
            The suspect's in-character response, safe to show to the player.
        """
        # Guard: respect the hard turn cap.
        if self.state.total_turns >= self.state.max_turns:
            logger.warning(
                "interrogate() called at turn %d but max_turns=%d — returning time-out message.",
                self.state.total_turns,
                self.state.max_turns,
            )
            return "(The detective has run out of time.)"

        profile       = self.get_current_suspect()
        suspect_agent = self.suspect_agents[self.current_suspect_id]
        prior_log     = self.conversation_logs[self.current_suspect_id]

        logger.info(
            "Turn %d/%d — interrogating %s (prior exchanges with this suspect: %d)",
            self.state.total_turns + 1,
            self.state.max_turns,
            profile.name,
            len(prior_log),
        )

        # Build history text (with optional compression for long sessions).
        history_text = self._build_history_text(profile, prior_log)

        suspect_prompt = (
            f"{history_text}"
            f"Detective's latest question:\n{player_message}\n\n"
            "Respond in character. Remember your public story and do not violate your redlines."
        )

        # Step 1: Raw suspect response.
        raw_resp = suspect_agent.run(suspect_prompt)
        raw_text: str = (
            raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
        )
        logger.debug("Raw suspect response: %d chars", len(raw_text))

        # Step 2: Conditional critique gate.
        safe_text = self._apply_critique_if_needed(
            player_message, profile, raw_text
        )

        # Step 3: Persist exchange and advance counters.
        self.conversation_logs[self.current_suspect_id].append(
            (player_message, safe_text)
        )
        self.state.add_turn(self.current_suspect_id)

        logger.info(
            "Turn complete — suspect=%s, total_turns_now=%d",
            profile.name,
            self.state.total_turns,
        )
        return safe_text

    def _apply_critique_if_needed(
        self,
        player_message: str,
        profile: SuspectProfile,
        raw_text: str,
    ) -> str:
        """
        Run the CritiqueAgent on raw_text only when a trigger keyword is present.

        The CritiqueAgent is an extra LLM call (utility-model size) that checks
        for and removes accidental leaks of the killer's identity, weapon, or
        hidden timeline. On average ~50% of turns don't need it.

        Args:
            player_message: The original detective question.
            profile:        The current suspect's SuspectProfile.
            raw_text:       The unfiltered response from the SuspectAgent.

        Returns:
            Either the original raw_text (clean) or a rewritten safe version.
        """
        raw_lower      = raw_text.lower()
        needs_critique = any(kw in raw_lower for kw in CRITIQUE_TRIGGERS)

        if not needs_critique:
            logger.debug(
                "Critique gate: PASS — no trigger keywords. Suspect=%s.", profile.name
            )
            return raw_text.strip()

        logger.info(
            "Critique gate: TRIGGERED for suspect=%s. Invoking CritiqueAgent.",
            profile.name,
        )

        critique_prompt = (
            f"PLAYER QUESTION:\n{player_message}\n\n"
            f"SUSPECT PROFILE:\n"
            f"  name: {profile.name}\n"
            f"  role: {profile.role}\n"
            f"  persona: {profile.persona}\n"
            f"  public_info: {profile.public_info}\n"
            f"  hard_redlines: {profile.hard_redlines}\n\n"
            f'RAW ANSWER FROM SUSPECT:\n"""{raw_text}"""\n\n'
            "Output ONLY the final, safe, in-character answer."
        )

        crit_resp = self.critique_agent.run(critique_prompt)
        safe_text: str = (
            crit_resp.content if hasattr(crit_resp, "content") else str(crit_resp)
        )
        logger.debug("CritiqueAgent rewrote response for suspect=%s.", profile.name)
        return safe_text.strip()

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------

    def _build_history_text(
        self,
        profile: SuspectProfile,
        prior_log: List[Tuple[str, str]],
    ) -> str:
        """
        Build the conversation-history prefix injected at the top of each
        suspect prompt.

        Strategy:
          - Turns 1 to GAME_CONFIG.context_compress_after:
              Pass the last history_verbatim_early exchanges verbatim.
          - Turns beyond that threshold, OR when the log is long:
              Older exchanges are compressed to one bullet each; the most
              recent history_verbatim_recent exchanges are kept verbatim.
              This keeps context window usage bounded without losing
              recent dialogue fidelity.

        Args:
            profile:   The suspect being addressed (for labelling).
            prior_log: Full conversation history for this suspect.

        Returns:
            A formatted string to prepend to the suspect's prompt,
            or an empty string if no prior exchanges exist.
        """
        cfg = GAME_CONFIG

        if not prior_log:
            return ""

        # Early-game: full verbatim history.
        if (
            self.state.total_turns < cfg.context_compress_after
            or len(prior_log) <= cfg.history_verbatim_recent
        ):
            recent = prior_log[-cfg.history_verbatim_early :]
            lines  = []
            for q, a in recent:
                lines.append(f"Detective: {q}")
                lines.append(f"You ({profile.name}): {a}")
            return (
                "PREVIOUS EXCHANGES IN THIS INTERROGATION SESSION:\n"
                + "\n".join(lines)
                + "\n\n"
            )

        # Late-game: compress older exchanges, keep recent ones verbatim.
        older  = prior_log[: -cfg.history_verbatim_recent]
        recent = prior_log[-cfg.history_verbatim_recent :]

        logger.debug(
            "Context compression active for suspect=%s: %d older exchanges summarised.",
            profile.name,
            len(older),
        )

        summary_points = [
            f"  - Asked about: '{q[:80]}' — you answered evasively / denied involvement."
            for q, _ in older
        ]
        summary_block = (
            "SUMMARY OF EARLIER EXCHANGES (condensed):\n"
            + "\n".join(summary_points)
        )

        recent_lines = []
        for q, a in recent:
            recent_lines.append(f"Detective: {q}")
            recent_lines.append(f"You ({profile.name}): {a}")
        recent_block = (
            "MOST RECENT EXCHANGES (full):\n" + "\n".join(recent_lines)
        )

        return summary_block + "\n\n" + recent_block + "\n\n"

    # ------------------------------------------------------------------
    # AI-detective interrogation planning
    # ------------------------------------------------------------------

    def plan_next_question(
        self, suspect_id: str, covered_topics: List[str]
    ) -> str:
        """
        Ask the InterrogationPlannerAgent to generate the single best next
        question for suspect_id given everything learned so far.

        The planner receives:
          - This suspect's prior exchanges (last planner_history_window turns).
          - Summarised intel from every OTHER suspect's recent answers.
          - A list of topics already covered with this suspect (to avoid repeats).
          - An auto-detected list of investigation gaps (weapon, motive, timeline).

        Args:
            suspect_id:     The suspect about to be questioned.
            covered_topics: Plain-text list of questions/themes already explored
                            with this suspect. Maintained by the AI agent caller.

        Returns:
            A single question string ending with '?', ready to pass to interrogate().
        """
        cfg       = GAME_CONFIG
        profile   = SUSPECTS[suspect_id]
        prior_log = self.conversation_logs[suspect_id]

        logger.debug(
            "Planning next question for suspect=%s, covered_topics=%d",
            profile.name,
            len(covered_topics),
        )

        # Format this suspect's recent exchanges for the planner.
        prior_lines: List[str] = []
        for q, a in prior_log[-cfg.planner_history_window :]:
            prior_lines.append(f"  Detective: {q}")
            prior_lines.append(f"  {profile.name}: {a}")
        prior_text = (
            "\n".join(prior_lines) if prior_lines else "  (No exchanges yet.)"
        )

        # Summarise what every OTHER suspect has said recently.
        other_intel_lines: List[str] = []
        for other_id, other_log in self.conversation_logs.items():
            if other_id == suspect_id or not other_log:
                continue
            other_name = SUSPECTS[other_id].name
            other_intel_lines.append(f"  [{other_name}]")
            for q, a in other_log[-cfg.planner_other_window :]:
                other_intel_lines.append(f"    Q: {q}")
                other_intel_lines.append(
                    f"    A: {a[:300]}{'...' if len(a) > 300 else ''}"
                )
        other_text = (
            "\n".join(other_intel_lines)
            if other_intel_lines
            else "  (No other suspects questioned yet — ask foundational questions.)"
        )

        # Detect evidence gaps from all answers given so far.
        gaps_text = self._detect_evidence_gaps(prior_log)

        covered_text = (
            "\n".join(f"  - {t}" for t in covered_topics)
            if covered_topics
            else "  (None yet — start with foundational questions.)"
        )

        planner_prompt = (
            f"CURRENT SUSPECT:\n"
            f"  Name    : {profile.name}\n"
            f"  Persona : {profile.persona}\n"
            f"  Public alibi: {profile.public_info}\n\n"
            f"PRIOR EXCHANGES WITH THIS SUSPECT:\n{prior_text}\n\n"
            f"COVERED TOPICS (do NOT repeat these):\n{covered_text}\n\n"
            f"INTEL FROM OTHER SUSPECTS:\n{other_text}\n\n"
            f"INVESTIGATION GAPS:\n{gaps_text}\n\n"
            "Now output the single best next question to ask this suspect."
        )

        resp     = self.interrogation_planner.run(planner_prompt)
        question: str = (
            resp.content if hasattr(resp, "content") else str(resp)
        )
        question = question.strip().strip('"').strip("'")

        # Safety: ensure well-formed question mark.
        if question and not question.endswith("?"):
            question += "?"

        logger.info("Planner generated question for %s: %r", profile.name, question)
        return question

    def _detect_evidence_gaps(
        self, prior_log: List[Tuple[str, str]]
    ) -> str:
        """
        Scan the suspect's prior answers for key evidence categories
        and return a formatted string listing what is still unknown.

        This heuristic checks for the presence of key topic words rather
        than doing full NLU — fast, deterministic, and good enough for
        directing the planner's attention.

        Args:
            prior_log: Full exchange history for the current suspect.

        Returns:
            A formatted multi-line string describing open investigation gaps.
        """
        all_answers = " ".join(a for _, a in prior_log).lower()
        gaps: List[str] = []

        if "candlestick" not in all_answers and "weapon" not in all_answers:
            gaps.append(
                "Weapon not confirmed — probe for physical objects seen that night."
            )
        if (
            "inheritance" not in all_answers
            and "money" not in all_answers
            and "motive" not in all_answers
        ):
            gaps.append(
                "Motive unclear — explore financial disputes or personal grievances."
            )
        if "library" not in all_answers:
            gaps.append(
                "Library (crime scene) not mentioned — ask if they were near it."
            )
        if "23" not in all_answers and "time" not in all_answers:
            gaps.append(
                "Precise timeline around 23:00-23:30 not established."
            )
        if not gaps:
            gaps.append(
                "All major gaps covered — probe for contradictions or emotional slips."
            )

        return "\n".join(f"  - {g}" for g in gaps)

    # ------------------------------------------------------------------
    # AI-detective deduction
    # ------------------------------------------------------------------

    def deduce_accusation(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Use the DeductionAgent to analyse all transcripts and return the most
        likely (suspect_id, weapon, motive) tuple.

        The agent is prompted to return a strict json block.
        We extract the JSON, parse it with json.loads(), and validate it through
        the DeductionResult Pydantic schema — giving a correctness guarantee
        without requiring any agno-specific response_model support.

        Returns:
            A 3-tuple of (suspect_id, weapon, motive) strings on success.
            Returns (None, None, None) if the agent's output cannot be parsed
            or validated. The caller is responsible for handling this explicitly
            rather than silently falling back to any particular suspect.

        Raises:
            Nothing — all exceptions are caught, logged at ERROR level, and
            signalled to the caller via the (None, None, None) sentinel.
        """
        transcript_lines: List[str] = []
        for sid, log in self.conversation_logs.items():
            if not log:
                continue
            transcript_lines.append(
                f"\n=== Interrogation: {SUSPECTS[sid].name} ==="
            )
            for i, (q, a) in enumerate(log, 1):
                transcript_lines.append(f"Q{i}: {q}")
                transcript_lines.append(f"A{i}: {a}")

        transcript_text = (
            "\n".join(transcript_lines)
            if transcript_lines
            else "No interrogations conducted."
        )

        logger.info(
            "DeductionAgent starting — total transcript lines: %d",
            len(transcript_lines),
        )

        deduction_prompt = (
            "Here are the full interrogation transcripts from the murder investigation.\n"
            "Analyse them carefully, then return your JSON accusation block.\n\n"
            f"{transcript_text}"
        )

        raw = "<no response>"
        try:
            resp = self.deduction_agent.run(deduction_prompt)
            raw  = resp.content if hasattr(resp, "content") else str(resp)

            # Extract JSON from json fences or bare braces.
            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
                json_str = json_match.group(0) if json_match else "{}"

            data   = json.loads(json_str)
            result = DeductionResult(**data)

            logger.info(
                "DeductionAgent concluded: suspect=%s, weapon=%r, motive=%r. "
                "Reasoning (first 120 chars): %s",
                result.suspect_id,
                result.weapon,
                result.motive,
                result.reasoning[:120],
            )
            return result.suspect_id, result.weapon, result.motive

        except Exception as exc:
            # Log at ERROR so this surfaces in any log aggregation system.
            # Return the explicit (None, None, None) failure sentinel.
            # The caller — app.py or cli.py — must decide what to show the
            # player. We must NOT silently pick a suspect here, as that would
            # produce a wrong accusation in any case where s1 is not guilty.
            logger.error(
                "DeductionAgent output parsing failed: %s. "
                "Raw response (first 300 chars): %r. "
                "Returning (None, None, None) — caller must handle.",
                exc,
                raw[:300],
                exc_info=True,
            )
            return None, None, None

    # ------------------------------------------------------------------
    # Accusation evaluation
    # ------------------------------------------------------------------

    def make_accusation(
        self,
        suspect_id: str,
        weapon: str,
        motive: str,
    ) -> Tuple[bool, int, str]:
        """
        Evaluate a player or AI-agent accusation against the ground truth.

        The SUSPECT PROFILES block in the judge's prompt is built dynamically
        from the SUSPECTS dict so that swapping case_data.py requires zero
        changes here.

        Steps:
          1. Compare accusation against CASE_FILE["truth"].
          2. Compute score via calculate_score().
          3. Build interrogation highlights (last 5 exchanges per suspect).
          4. Build suspect profiles block dynamically from SUSPECTS.
          5. Send everything to the AccusationAgent to produce the narrative.
          6. Persist result in GameState.

        Args:
            suspect_id: The accused suspect's ID string (e.g. "s1").
            weapon:     The claimed murder weapon (must be in VALID_WEAPONS).
            motive:     The claimed motive (must be in VALID_MOTIVES).

        Returns:
            Tuple of (game_won: bool, score: int, evaluation_text: str).
        """
        truth = CASE_FILE["truth"]

        correct_suspect = suspect_id == truth["culprit_id"]
        correct_weapon  = weapon.lower().strip() == truth["method"].lower().strip()
        correct_motive  = motive.lower().strip() == truth["motive"].lower().strip()

        score = calculate_score(
            correct_suspect,
            correct_weapon,
            correct_motive,
            self.state.total_turns,
        )

        logger.info(
            "Accusation received — suspect=%r, weapon=%r, motive=%r | "
            "correct=(%s, %s, %s) | score=%d",
            suspect_id, weapon, motive,
            correct_suspect, correct_weapon, correct_motive,
            score,
        )

        # Build interrogation highlights (last 5 Q&A per suspect).
        highlights = self._build_interrogation_highlights()

        accused_name = (
            SUSPECTS[suspect_id].name if suspect_id in SUSPECTS else "Unknown"
        )
        culprit_name = SUSPECTS[truth["culprit_id"]].name

        # Build suspect profiles block dynamically from the SUSPECTS dict.
        # This replaces the old hardcoded string that named Lydia, Marcus, and
        # Eleanor literally — making make_accusation() fully case-agnostic.
        suspect_profiles_lines: List[str] = []
        for sid, profile in SUSPECTS.items():
            is_culprit = sid == truth["culprit_id"]
            marker     = " <- TRUE CULPRIT" if is_culprit else ""
            suspect_profiles_lines.append(
                f"  {sid}: {profile.name} ({profile.role.upper()}){marker}\n"
                f"      Public alibi : {profile.public_info}\n"
                f"      Hidden truth : {profile.secret_info}"
            )
        suspect_profiles_block = "\n".join(suspect_profiles_lines)

        eval_prompt = (
            f"PLAYER'S ACCUSATION:\n"
            f"  Suspect : {suspect_id} ({accused_name})\n"
            f"  Weapon  : {weapon}\n"
            f"  Motive  : {motive}\n\n"
            f"THE TRUTH:\n"
            f"  Culprit : {truth['culprit_id']} ({culprit_name})\n"
            f"  Weapon  : {truth['method']}\n"
            f"  Motive  : {truth['motive']}\n"
            f"  Timeline:\n"
            + "\n".join(f"    {t}" for t in truth["timeline"])
            + f"\n\nCORRECTNESS:\n"
            f"  Suspect correct : {correct_suspect}\n"
            f"  Weapon correct  : {correct_weapon}\n"
            f"  Motive correct  : {correct_motive}\n\n"
            f"COMPUTED SCORE: {score}/100\n\n"
            f"GAME STATISTICS:\n"
            f"  Total turns used     : {self.state.total_turns}\n"
            f"  Suspects interviewed : {len(self.state.suspects_interviewed)}/{len(SUSPECTS)}\n\n"
            f"SUSPECT PROFILES:\n{suspect_profiles_block}\n\n"
            f"INTERROGATION HIGHLIGHTS:\n{highlights}\n\n"
            "Produce the full CASE RESOLUTION evaluation now."
        )

        resp = self.accusation_agent.run(eval_prompt)
        eval_text: str = (
            resp.content if hasattr(resp, "content") else str(resp)
        )

        # Persist result.
        self.state.accusation_made = True
        self.state.game_won        = correct_suspect
        self.state.final_score     = score

        logger.info(
            "Accusation evaluation complete — game_won=%s, final_score=%d",
            correct_suspect,
            score,
        )
        return correct_suspect, score, eval_text.strip()

    def _build_interrogation_highlights(self) -> str:
        """
        Build a compact summary of interrogation highlights for the judge.

        Returns the last 5 (Q, A) pairs per suspect, with answers truncated
        to 200 characters to keep the eval prompt a manageable size.

        Returns:
            Multi-line string of highlights, or a 'no data' message.
        """
        highlight_lines: List[str] = []
        for sid, log in self.conversation_logs.items():
            if not log:
                continue
            name = SUSPECTS[sid].name
            highlight_lines.append(f"\n--- {name} ({len(log)} exchanges) ---")
            for q, a in log[-5:]:
                highlight_lines.append(f"  Q: {q}")
                truncated = f"{a[:200]}{'...' if len(a) > 200 else ''}"
                highlight_lines.append(f"  A: {truncated}")

        return (
            "\n".join(highlight_lines)
            if highlight_lines
            else "No interrogations recorded."
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the game for a new playthrough.

        Clears GameState, conversation logs, and rebuilds all agents from
        scratch so no prior session context can leak into the new game.
        """
        logger.info("Game reset requested — clearing state and rebuilding agents.")
        self.state.reset()
        self.conversation_logs  = {sid: [] for sid in SUSPECTS}
        self.current_suspect_id = "s1"
        self._build_all_agents()
        logger.info("Game reset complete.")
