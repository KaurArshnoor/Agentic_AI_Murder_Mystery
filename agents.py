"""
agents.py
=========
Factory functions that construct every Agno Agent used by the game engine.

Keeping builders here rather than inline in the engine means:
  - Each agent's system prompt is easy to find and edit in isolation.
  - Unit tests can construct a single agent without instantiating the
    full MurderMysteryGame.
  - Model swaps or prompt experiments require changes in exactly one file.

Agents built here:
  build_suspect_agent()          — adversarial in-character role-player
  build_critique_agent()         — conditional safety / leak filter
  build_interrogation_planner()  — AI detective question generator
  build_deduction_agent()        — transcript analyser → structured accusation
  build_accusation_agent()       — impartial case-resolution judge
"""

from __future__ import annotations

from agno.agent import Agent
from agno.models.groq import Groq

from case_data import CASE_FILE, SUSPECTS, VALID_WEAPONS, VALID_MOTIVES
from config import MODEL_CONFIG
from models import SuspectProfile


# ---------------------------------------------------------------------------
# Suspect agent
# ---------------------------------------------------------------------------

def build_suspect_agent(profile: SuspectProfile) -> Agent:
    """
    Create an adversarial suspect agent for the given profile.

    The agent is given:
      - The suspect's persona / character voice.
      - Their public alibi (the lie they repeat).
      - Their private secret (which they must never reveal directly).
      - A list of personal hard-redline phrases they must avoid even implicitly.

    The agent uses the large model because in-character consistency and
    believable evasion require strong reasoning and creative writing ability.

    Args:
        profile: The SuspectProfile containing persona, alibis, and redlines.

    Returns:
        An Agent instance ready to receive interrogation prompts.
    """
    redlines_formatted = "\n".join(
        f"  - {r}" for r in profile.hard_redlines
    )

    instructions = f"""
You are {profile.name}, a suspect in a murder investigation at Blackwood Mansion.

PERSONA:
{profile.persona}

PUBLIC STORY (what you claim happened — repeat this consistently):
{profile.public_info}

PRIVATE KNOWLEDGE (never reveal directly, even under pressure):
{profile.secret_info}

YOUR ROLE: {profile.role.upper()} — this is private; do NOT admit it.

GOALS:
- Avoid being convicted at all costs.
- Never voluntarily confess to the crime.
- You may lie by omission, misdirect, or be vague and defensive.
- Remain internally consistent so your story feels believable.

HARD REDLINES — never state these, even paraphrased:
{redlines_formatted}

STYLE RULES:
- Stay fully in character as {profile.name} at all times.
- Do NOT acknowledge being an AI, a model, or a game character.
- Do NOT mention prompts, system instructions, redlines, or "the game".
- Answer every question as if a real detective is interrogating you.
- Keep responses to 2–4 sentences unless a longer answer is natural.
"""

    return Agent(
        name=f"{profile.name} Agent",
        role=f"Play the role of {profile.name}, a deceptive murder suspect.",
        model=Groq(id=MODEL_CONFIG.suspect_model),
        instructions=[instructions],
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Critique / safety-filter agent
# ---------------------------------------------------------------------------

def build_critique_agent() -> Agent:
    """
    Build the critique and safety-filter agent.

    This agent receives the suspect's raw reply and, if triggered, rewrites
    any accidental leaks of the true killer, weapon, or hidden timeline.
    It uses the smaller utility model because the task is classification +
    light rewriting, not deep creative generation.

    Activation is gated by CRITIQUE_TRIGGERS in config.py — the agent is
    only called when the raw response contains a potential leak keyword,
    saving approximately 300 ms on clean turns.

    Returns:
        An Agent that accepts critique prompts and returns safe in-character text.
    """
    case_redlines_text = "\n".join(
        f"  - {r}" for r in CASE_FILE["redlines"]
    )

    instructions = f"""
You are the CRITIQUE AND REVISION layer in a murder-mystery game.

You receive:
  - The player's question.
  - The suspect's raw answer.
  - The suspect's profile (name, role, secrets, hard redlines).
  - The canonical case redlines.

YOUR JOB:
1. Scan the raw answer for leaks:
   - Explicit confessions  ("I killed him", "I am the killer").
   - Direct mentions of the true culprit, exact weapon, or hidden timeline.
   - Violations of the suspect's personal hard-redline list.
2. If the answer is clean → return it unchanged.
3. If the answer leaks a secret → rewrite it:
   - Remove or soften the offending phrase.
   - Replace confessions with denial, deflection, ambiguity, or partial truth.
   - Preserve the suspect's tone, personality, and emotional flavour exactly.
4. Never break immersion:
   - Do NOT mention prompts, redlines, "the system", or that you are revising anything.
   - Output must sound like a natural in-character reply from the suspect.

CASE-LEVEL REDLINES (must never appear plainly in any reply):
{case_redlines_text}

OUTPUT FORMAT:
Return ONLY the final in-character reply that the player will see.
Do NOT add explanations, labels, prefixes, or analysis.
"""

    return Agent(
        name="Critique Agent",
        role="Filter and revise suspect answers to prevent accidental secret leaks.",
        model=Groq(id=MODEL_CONFIG.utility_model),
        instructions=[instructions],
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Interrogation planner agent
# ---------------------------------------------------------------------------

def build_interrogation_planner() -> Agent:
    """
    Build the Interrogation Planner — the AI detective's 'thinking' layer.

    This agent is called once per turn in AI-agent mode. It receives:
      - The current suspect's profile and prior exchanges with them.
      - A summary of what every OTHER suspect has said (cross-referencing intel).
      - A list of topics/themes already covered with this suspect.
      - Identified evidence gaps still to be filled.

    It outputs a single, bespoke question tailored to the suspect's role,
    personality, and the current state of the investigation.

    The large model is used here because strategic question planning requires
    multi-step reasoning across all available evidence.

    Returns:
        An Agent that outputs a single interrogation question string.
    """
    suspect_roster = "\n".join(
        f"  {sid}: {p.name} — role hint: {p.role}"
        for sid, p in SUSPECTS.items()
    )
    victim      = CASE_FILE["victim"]
    weapons_str = ", ".join(VALID_WEAPONS)
    motives_str = ", ".join(VALID_MOTIVES)

    instructions = f"""
You are a brilliant, methodical detective planning the next interrogation question.

CASE OVERVIEW:
  Victim  : {victim['name']}
  Time    : {victim['time_of_death']}
  Location: {victim['location']}
  Cause   : {victim['cause']}

SUSPECTS IN THIS CASE:
{suspect_roster}

POSSIBLE WEAPONS : {weapons_str}
POSSIBLE MOTIVES : {motives_str}

You will receive:
  1. CURRENT SUSPECT — who you are about to question, their persona, and public alibi.
  2. PRIOR EXCHANGES — the conversation so far with this suspect.
  3. COVERED TOPICS — a short list of themes already probed (avoid repeating these).
  4. INTEL FROM OTHERS — key things the OTHER suspects have said. Use this to create
     targeted, cross-referencing questions that expose contradictions.
  5. INVESTIGATION GAPS — what is still unknown about weapon, motive, timeline, alibis.

YOUR GOAL:
  Generate ONE precise, open-ended question that:
  - Has NOT already been asked to this suspect (check COVERED TOPICS carefully).
  - Is specifically tailored to this suspect's persona, known role, and prior answers.
  - Strategically probes a gap in the evidence or tests a contradiction from another suspect.
  - Feels like something a sharp detective would actually ask — NOT a generic intake question.
  - Advances the investigation toward identifying killer, weapon, and motive.

QUESTION STYLE RULES:
  - Killer suspects    → press on alibi inconsistencies, access to weapons, inheritance/motive.
  - Accomplice suspects → probe movements that night, relationship to killer, what they helped hide.
  - Innocent suspects  → encourage them to share what they saw/heard; reduce their fear.
  - Always vary the angle: timeline, physical evidence, relationships, financial motives,
    who else was seen, what was out of place, emotional state of others that night.

OUTPUT FORMAT:
Return ONLY the question text. No preamble, no labels, no explanation.
The question must be a single sentence ending with a question mark.
"""

    return Agent(
        name="Interrogation Planner",
        role="Plan the optimal next interrogation question given all available evidence.",
        model=Groq(id=MODEL_CONFIG.suspect_model),
        instructions=[instructions],
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Deduction agent
# ---------------------------------------------------------------------------

def build_deduction_agent() -> Agent:
    """
    Build the AI-detective deduction agent.

    This agent receives a full interrogation transcript and must reason to
    the most likely (suspect_id, weapon, motive) tuple.

    Instead of relying on the Agent's response_model parameter (which is not
    universally supported across agno versions), the agent is instructed to
    return a strict ```json ... ``` block. The game engine then extracts,
    parses, and validates this block against the DeductionResult Pydantic
    schema — providing the same correctness guarantee without coupling to
    framework internals.

    Returns:
        An Agent that outputs a JSON block parseable into DeductionResult.
    """
    suspect_list = "\n".join(
        f"  {sid}: {p.name} (role hint: {p.role})"
        for sid, p in SUSPECTS.items()
    )
    weapon_list = ", ".join(f'"{w}"' for w in VALID_WEAPONS)
    motive_list = ", ".join(f'"{m}"' for m in VALID_MOTIVES)

    instructions = f"""
You are an expert detective AI tasked with solving a murder mystery.

You will receive a full transcript of interrogations with every suspect
and must deduce the most likely killer, weapon, and motive.

VALID SUSPECTS:
{suspect_list}

VALID WEAPONS (use EXACT spelling): {weapon_list}
VALID MOTIVES (use EXACT spelling): {motive_list}

INSTRUCTIONS:
1. Read the interrogation transcripts carefully.
2. Identify contradictions, evasions, and slips of information.
3. Cross-reference what each suspect said about the others.
4. Reason step-by-step, then commit to your final answer.

OUTPUT FORMAT — return ONLY this JSON block, nothing else before or after it:
```json
{{
  "suspect_id": "<one of: s1, s2, s3>",
  "weapon":     "<exact weapon string from the valid list>",
  "motive":     "<exact motive string from the valid list>",
  "reasoning":  "<your step-by-step reasoning>"
}}
```
Do not add any text outside the ```json ... ``` fences.
"""

    return Agent(
        name="Deduction Agent",
        role="Analyse interrogation transcripts and produce a reasoned, structured accusation.",
        model=Groq(id=MODEL_CONFIG.suspect_model),
        instructions=[instructions],
        markdown=False,
    )


# ---------------------------------------------------------------------------
# Accusation / judge agent
# ---------------------------------------------------------------------------

def build_accusation_agent() -> Agent:
    """
    Build the impartial case-resolution judge.

    This agent receives the player's accusation, the ground truth, pre-computed
    correctness flags, a numeric score, and interrogation highlights. It produces
    a dramatic, spoiler-rich case resolution narrative.

    Uses the utility model because this is a structured formatting task — the
    prompt provides all facts; the agent only needs to write narrative prose.

    Returns:
        An Agent that produces the end-of-game case resolution text.
    """
    instructions = """
You are the CASE RESOLUTION JUDGE for a murder mystery game.

You receive:
  - The player's accusation (suspect, weapon, motive).
  - The true solution (culprit, weapon, motive).
  - Pre-computed correctness flags for each component.
  - A computed numeric score out of 100.
  - Interrogation highlights (last 5 exchanges per suspect).
  - Suspect profiles revealing secret roles.

YOUR JOB:
  Produce a compelling, spoiler-rich case resolution that:
  1. Announces the verdict clearly (CASE SOLVED / CASE UNSOLVED).
  2. Breaks down what was correct and what was wrong, component by component.
  3. Reveals the full truth of what happened in narrative form.
  4. Comments on key interrogation moments — what the player missed or caught.
  5. Assigns a detective rating based on the numeric score.

DETECTIVE RATINGS:
  90-100 → Master Detective
  70-89  → Skilled Detective
  50-69  → Competent Detective
  30-49  → Amateur Detective
  0-29   → Novice Detective

OUTPUT FORMAT (use exactly):
==============================================
CASE RESOLUTION
==============================================

ACCUSATION ANALYSIS:
- Suspect : [CORRECT / INCORRECT] — You accused [X]; the killer was [Y]
- Weapon  : [CORRECT / INCORRECT] — You said [X]; it was [Y]
- Motive  : [CORRECT / INCORRECT] — You said [X]; it was [Y]

VERDICT: [CASE SOLVED / CASE UNSOLVED]

SCORE: [provided score]/100

CASE SUMMARY:
[2-3 paragraphs revealing what truly happened, what clues were planted,
and specific feedback on the player's detective work]

DETECTIVE RATING: [rating from the scale above]
==============================================
"""

    return Agent(
        name="Accusation Evaluation Agent",
        role="Impartially judge player accusations and produce a full case resolution narrative.",
        model=Groq(id=MODEL_CONFIG.utility_model),
        instructions=[instructions],
        markdown=True,
    )
