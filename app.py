"""
app.py
======
Streamlit web UI for AI Murder Mystery: The Blackwood Mansion.

Responsibilities:
  - Configure and render the Streamlit page (layout, dark-noir theme).
  - Manage session state initialisation and reset.
  - Render sidebar components (suspect selector, game status).
  - Render main-panel components (case briefing, chat interface, accusation form).
  - Orchestrate the multi-phase AI agent investigation flow.

This file contains only UI logic. All game logic lives in game_engine.py,
all agents in agents.py, all narrative data in case_data.py, and all shared
utilities in ui_helpers.py.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

# Load .env before any game code runs so GROQ_API_KEY is available.
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
#
# basicConfig is called here — the Streamlit entry point — so it runs exactly
# once per process regardless of how many times Streamlit reruns the script.
# All modules under "murder_mystery.*" emit to this handler automatically via
# Python's hierarchical logger namespace.
#
# In production, swap StreamHandler for a FileHandler, RotatingFileHandler,
# or a structured JSON handler (e.g. python-json-logger) without touching any
# other file.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("murder_mystery.app")

from case_data import CASE_FILE, SUSPECTS, VALID_MOTIVES, VALID_WEAPONS
from config import GAME_CONFIG
from game_engine import MurderMysteryGame
from ui_helpers import build_css, is_duplicate_question, normalize_to_set


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Murder Mystery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"<style>{build_css()}</style><div class='vignette'></div>",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    """
    Initialise all Streamlit session state variables on first run.

    Uses a defaults dict so new keys can be added in one place without
    multiple scattered `if key not in st.session_state` guards.

    Sidebar placeholder keys (prefixed _sb_) are set to None here and
    populated by render_game_status() / render_suspect_selector() on
    each render pass. run_ai_agent_interrogation() writes into them on
    every question turn for real-time sidebar updates without full reruns.
    """
    defaults: dict = {
        "game":              MurderMysteryGame(),
        "messages":          {sid: [] for sid in SUSPECTS},
        "current_suspect":   "s1",
        "game_over":         False,
        "accusation_result": None,
        "notes":             {sid: "" for sid in SUSPECTS},
        "general_notes":     "",
        "game_mode":         "manual",   # "manual" | "ai_agent"
        "ai_agent_active":   False,
        # Sidebar live-update placeholders.
        "_sb_progress":        None,
        "_sb_remaining":       None,
        "_sb_interviewed":     None,
        "_sb_suspect_cards":   None,
        "_sb_current_suspect": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_game() -> None:
    """
    Fully reset the game and all UI state for a new investigation.

    Rebuilds the game engine (which clears agent memory) and wipes all
    session state variables that carry per-game information.
    """
    st.session_state.game               = MurderMysteryGame()
    st.session_state.messages           = {sid: [] for sid in SUSPECTS}
    st.session_state.current_suspect    = "s1"
    st.session_state.game_over          = False
    st.session_state.accusation_result  = None
    st.session_state.notes              = {sid: "" for sid in SUSPECTS}
    st.session_state.general_notes      = ""
    st.session_state.game_mode          = "manual"
    st.session_state.ai_agent_active    = False
    st.session_state._sb_progress       = None
    st.session_state._sb_remaining      = None
    st.session_state._sb_interviewed    = None
    st.session_state._sb_suspect_cards  = None
    st.session_state._sb_current_suspect = None


# ============================================================
# SIDEBAR COMPONENTS
# ============================================================

def render_suspect_selector() -> None:
    """
    Render the suspect list and current-subject label in the sidebar.

    Each suspect card and the "Currently interrogating" label are wrapped in
    st.sidebar.empty() placeholders stored in session state. During AI agent
    mode, _update_suspect_sidebar() pushes fresh content into those placeholders
    on every question turn, giving real-time updates without a full page rerun.
    """
    st.sidebar.markdown(
        '<div class="sidebar-header">👥 SUSPECTS</div>', unsafe_allow_html=True
    )
    st.sidebar.markdown("")

    # One empty placeholder per suspect card.
    card_placeholders: dict = {sid: st.sidebar.empty() for sid in SUSPECTS}
    st.session_state._sb_suspect_cards = card_placeholders

    st.sidebar.markdown("---")
    st.session_state._sb_current_suspect = st.sidebar.empty()

    # Initial render.
    _update_suspect_sidebar(
        active_sid=st.session_state.current_suspect,
        game_state=st.session_state.game.state,
    )


def _update_suspect_sidebar(active_sid: str, game_state) -> None:
    """
    Push a fresh render of all suspect cards and the current-subject label
    into their sidebar placeholders.

    Called once on initial render and then live from _ask_one() whenever the
    AI agent switches to a different suspect.

    Args:
        active_sid:  The suspect currently being questioned.
        game_state:  The live GameState object (for turn counts).
    """
    card_placeholders = st.session_state._sb_suspect_cards
    current_ph        = st.session_state._sb_current_suspect

    if card_placeholders is None or current_ph is None:
        return  # Placeholders not yet created.

    for sid, suspect in SUSPECTS.items():
        ph          = card_placeholders[sid]
        turns       = game_state.turns_per_suspect.get(sid, 0)
        interviewed = sid in game_state.suspects_interviewed

        if sid == active_sid:
            icon, status, colour = (
                "🔴", f"Questioning now… ({turns} so far)", "#8B0000"
            )
        elif interviewed:
            icon, status, colour = (
                "🔍",
                f"{turns} question{'s' if turns != 1 else ''} asked",
                "#555",
            )
        else:
            icon, status, colour = "⬜", "Not yet questioned", "#444"

        if st.session_state.ai_agent_active:
            # In AI mode render an informational card (no interactive button).
            ph.markdown(
                f"<div style='"
                f"background:#1a1a1a;border:1px solid {colour};border-radius:8px;"
                f"padding:10px 12px;margin:4px 0;font-family:Courier Prime,monospace;'>"
                f"<span style='color:{colour};font-size:13px;'>{icon} <b>{suspect.name}</b></span><br>"
                f"<span style='color:#666;font-size:11px;'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            # Manual mode — render a real clickable button.
            with ph:
                if st.button(
                    f"{icon} {suspect.name}\n{status}",
                    key=f"suspect_{sid}",
                    use_container_width=True,
                    disabled=st.session_state.game_over,
                ):
                    st.session_state.current_suspect = sid
                    st.session_state.game.switch_suspect(sid)
                    st.rerun()

    # "Currently interrogating" label.
    current = SUSPECTS[active_sid]
    if st.session_state.ai_agent_active:
        current_ph.markdown(
            f"<div style='font-family:Courier Prime,monospace;padding:6px 0;'>"
            f"<span style='color:#666;font-size:12px;'>🤖 AI agent interrogating:</span><br>"
            f"<span style='color:#8B0000;font-size:18px;font-family:Special Elite,cursive;'>"
            f"🎭 {current.name}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        current_ph.markdown(f"**Currently interrogating:**\n\n### 🎭 {current.name}")


def render_game_status() -> None:
    """
    Render the turn counter and investigation progress section in the sidebar.

    Uses st.sidebar.empty() placeholders stored in session state so that
    run_ai_agent_interrogation() can push live updates on every question
    without triggering a full page rerun.
    """
    state = st.session_state.game.state

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="sidebar-header">📊 INVESTIGATION</div>',
        unsafe_allow_html=True,
    )

    # Create placeholders and store references.
    st.session_state._sb_progress    = st.sidebar.empty()
    st.session_state._sb_remaining   = st.sidebar.empty()
    st.session_state._sb_interviewed = st.sidebar.empty()
    _sb_warning                      = st.sidebar.empty()

    _render_sidebar_stats(
        state,
        st.session_state._sb_progress,
        st.session_state._sb_remaining,
        st.session_state._sb_interviewed,
        _sb_warning,
    )


def _render_sidebar_stats(
    state, ph_progress, ph_remaining, ph_interviewed, ph_warning
) -> None:
    """
    Write current game stats into the provided sidebar placeholder widgets.

    Args:
        state:          Current GameState.
        ph_progress:    Placeholder for the progress bar.
        ph_remaining:   Placeholder for the remaining-questions counter.
        ph_interviewed: Placeholder for the suspects-questioned counter.
        ph_warning:     Placeholder for time-running-out warnings.
    """
    progress  = state.total_turns / max(1, state.max_turns)
    remaining = state.max_turns - state.total_turns

    ph_progress.progress(progress)
    ph_remaining.markdown(f"**Questions remaining:** {remaining}")
    ph_interviewed.markdown(
        f"**Suspects questioned:** {len(state.suspects_interviewed)} / 3"
    )

    if 0 < remaining <= 5 and not state.accusation_made:
        ph_warning.error(f"⚠️ Only {remaining} questions left!")
    elif remaining == 0 and not state.accusation_made:
        ph_warning.error("🚨 Time's up! Make your accusation!")


# ============================================================
# MAIN-PANEL COMPONENTS
# ============================================================

def render_case_briefing() -> None:
    """Render the top-level case file card above the chat interface."""
    v = CASE_FILE["victim"]
    st.markdown("""
    <div class="case-file">
        <h3>📁 CLASSIFIED CASE FILE</h3>
        <p style="color: #666; font-size: 12px;">BLACKWOOD MANSION HOMICIDE — FILE #1947</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**VICTIM:** {v['name']}")
        st.markdown(f"**TIME OF DEATH:** {v['time_of_death']}")
    with col2:
        st.markdown(f"**LOCATION:** {v['location']}")
        st.markdown(f"**CAUSE:** {v['cause']}")

    st.markdown("---")
    st.markdown(
        "*Your mission: Interrogate the suspects. "
        "Discover WHO committed the murder, with WHAT weapon, and WHY.*"
    )


def render_chat_interface() -> None:
    """
    Render the main interrogation chat interface for manual play.

    Displays:
      - A header showing who is currently being interrogated.
      - A scrollable chat history (400 px fixed height).
      - Six suggested quick-question chips.
      - A free-text chat input.
    """
    current_suspect = SUSPECTS[st.session_state.current_suspect]

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, transparent, #1a1a1a, transparent);
                padding: 10px; text-align: center;">
        <span style="font-family: 'Special Elite', cursive; font-size: 24px; color: #8B0000;">
            🎭 INTERROGATION ROOM
        </span><br>
        <span style="font-family: 'Courier Prime', monospace; color: #666;">
            Subject: {current_suspect.name}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Scrollable chat container.
    chat_container = st.container(height=400)
    with chat_container:
        messages = st.session_state.messages[st.session_state.current_suspect]
        if not messages:
            st.markdown(f"""
            <div class="typewriter-text" style="text-align: center; padding: 50px; color: #666;">
                <em>*{current_suspect.name} sits across the table,
                the dim light casting shadows across their face...*</em>
            </div>
            """, unsafe_allow_html=True)

        for msg in messages:
            if msg["role"] == "user":
                st.chat_message("user", avatar="🕵️").markdown(msg["content"])
            else:
                st.chat_message("assistant", avatar="🎭").markdown(
                    f'*{msg["content"]}*'
                )

    if st.session_state.game_over:
        return

    state = st.session_state.game.state
    if state.total_turns >= state.max_turns:
        st.error("🚨 No questions remaining! You must make your accusation now.")
        return

    # Quick-question suggestion chips.
    _render_suggestion_chips()

    # Free-text input.
    user_input = st.chat_input(
        placeholder=(
            f"Interrogate {current_suspect.name}… "
            "e.g., 'Where were you at the time of the murder?'"
        )
    )
    if user_input:
        _submit_question(user_input)
        st.rerun()


def _render_suggestion_chips() -> None:
    """
    Render a 2-row grid of pre-written quick-question buttons.

    Clicking any chip submits the question directly, bypassing the text input.
    """
    suggested_prompts = [
        "Where were you at the time of the murder?",
        "How do you know the victim?",
        "Did you see anyone suspicious that night?",
        "Can anyone corroborate your alibi?",
        "Why would someone want to hurt the victim?",
        "Did you have any disagreements with the victim?",
    ]
    st.markdown("**Suggested questions:**")
    chunk_size = 3
    for i in range(0, len(suggested_prompts), chunk_size):
        row  = suggested_prompts[i : i + chunk_size]
        cols = st.columns(len(row))
        for j, prompt in enumerate(row):
            btn_key = f"suggest_{st.session_state.current_suspect}_{i + j}"
            if cols[j].button(prompt, key=btn_key, use_container_width=True):
                _submit_question(prompt)
                st.rerun()


def _submit_question(question: str) -> None:
    """
    Send a question to the current suspect and store the exchange in session state.

    Args:
        question: The question string from the text input or a suggestion chip.
    """
    sid             = st.session_state.current_suspect
    current_suspect = SUSPECTS[sid]

    st.session_state.messages[sid].append({"role": "user", "content": question})

    with st.spinner(f"*{current_suspect.name} considers your question…*"):
        response = st.session_state.game.interrogate(question)

    st.session_state.messages[sid].append({"role": "assistant", "content": response})


def render_notes_section() -> None:
    """
    Render the detective's tabbed notes panel below the chat interface.

    Provides a General tab and one tab per suspect for the player to record
    observations and theories. Notes persist in session state for the
    duration of the investigation.
    """
    st.markdown("---")
    st.markdown(
        '<div class="notes-header">📝 DETECTIVE\'S NOTES</div>',
        unsafe_allow_html=True,
    )

    tab_labels = [
        "📋 General Notes",
        f"🎭 {SUSPECTS['s1'].name}",
        f"🎭 {SUSPECTS['s2'].name}",
        f"🎭 {SUSPECTS['s3'].name}",
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

    with tab1:
        st.session_state.general_notes = st.text_area(
            "General observations and theories:",
            value=st.session_state.general_notes,
            height=150,
            placeholder="Write your theories here…",
            key="general_notes_input",
        )
    for tab, sid in zip([tab2, tab3, tab4], ["s1", "s2", "s3"]):
        with tab:
            st.session_state.notes[sid] = st.text_area(
                f"Notes on {SUSPECTS[sid].name}:",
                value=st.session_state.notes[sid],
                height=150,
                placeholder=(
                    f"What did {SUSPECTS[sid].name} reveal? "
                    "Any suspicious behaviour?"
                ),
                key=f"notes_{sid}",
            )


def render_accusation_form() -> None:
    """
    Render the accusation form in an expander (manual mode only).

    The form is hidden until the player explicitly opens it, preventing
    accidental early accusations. It disables itself after submission.
    """
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <span style="font-family: 'Special Elite', cursive; font-size: 28px; color: #8B0000;">
            ⚖️ MAKE YOUR ACCUSATION
        </span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.game_over:
        st.info("The case is closed. Click 'New Case' to investigate another mystery.")
        return

    with st.expander("🔓 Ready to name the killer? Click here…", expanded=False):
        st.markdown("""
        <p style="font-family: 'Courier Prime', monospace; color: #888; text-align: center;">
            Choose wisely, detective. You only get one chance.
        </p>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**THE KILLER**")
            accused = st.selectbox(
                "Who committed the murder?",
                options=list(SUSPECTS.keys()),
                format_func=lambda x: SUSPECTS[x].name,
                label_visibility="collapsed",
            )
        with col2:
            st.markdown("**THE WEAPON**")
            weapon = st.selectbox(
                "What was the murder weapon?",
                options=VALID_WEAPONS,
                label_visibility="collapsed",
            )
        with col3:
            st.markdown("**THE MOTIVE**")
            motive = st.selectbox(
                "Why did they do it?",
                options=VALID_MOTIVES,
                label_visibility="collapsed",
            )

        st.markdown("")
        st.warning("⚠️ This is your FINAL accusation. There is no going back.")

        if st.button("🔨 I ACCUSE…", type="primary", use_container_width=True):
            with st.spinner("*The room falls silent as the evidence is reviewed…*"):
                won, score, eval_text = st.session_state.game.make_accusation(
                    accused, weapon, motive
                )
            st.session_state.game_over          = True
            st.session_state.accusation_result  = {
                "won": won, "score": score, "eval_text": eval_text, "ai_agent": False,
            }
            st.rerun()


def render_game_result() -> None:
    """
    Render the end-of-game result screen.

    Shows the verdict banner (CASE SOLVED / CASE UNSOLVED), the numeric score,
    a full interrogation transcript expander, and the judge's case resolution report.
    """
    if not st.session_state.accusation_result:
        return

    result = st.session_state.accusation_result
    st.markdown("---")

    if result["won"]:
        st.balloons()
        st.markdown("""
        <div style="text-align: center; padding: 30px;">
            <span style="font-family: 'Special Elite', cursive; font-size: 48px; color: #228B22;">
                🎉 CASE SOLVED
            </span><br><br>
            <span style="font-family: 'Courier Prime', monospace; color: #666;">
                Justice has been served. The killer is behind bars.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 30px;">
            <span style="font-family: 'Special Elite', cursive; font-size: 48px; color: #8B0000;">
                ❌ CASE UNSOLVED
            </span><br><br>
            <span style="font-family: 'Courier Prime', monospace; color: #666;">
                The killer walks free… for now.
            </span>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="score-display">{result['score']}/100</div>
        <p style="text-align: center; font-family: 'Courier Prime', monospace; color: #666;">
            DETECTIVE RATING
        </p>
        """, unsafe_allow_html=True)

    is_ai = result.get("ai_agent", False)
    label = "🕵️ Agent" if is_ai else "🕵️ You"

    with st.expander("📝 Interrogation Transcript", expanded=False):
        for sid in ["s1", "s2", "s3"]:
            messages = st.session_state.messages[sid]
            if messages:
                st.markdown(f"### 🎭 {SUSPECTS[sid].name}")
                for msg in messages:
                    if msg["role"] == "user":
                        st.markdown(f"**{label}:** {msg['content']}")
                    else:
                        st.markdown(f"**{SUSPECTS[sid].name}:** {msg['content']}")
                st.markdown("---")

    with st.expander("📋 Case Resolution Report", expanded=True):
        st.markdown(result["eval_text"])

    st.markdown("")
    if st.button("🔄 NEW CASE", type="primary", use_container_width=True):
        reset_game()
        st.rerun()


# ============================================================
# AI AGENT ORCHESTRATION
# ============================================================

def run_ai_agent_interrogation() -> None:
    """
    Orchestrate the full AI-agent investigation and accusation flow.

    The agent runs in four sequential phases without user interaction:

    Phase 1 — Initial interrogation
        Each suspect is questioned for up to GAME_CONFIG.ai_max_first_pass turns.
        Questions are generated by the InterrogationPlanner, which has access to
        all prior exchanges and cross-suspect intel.

    Phase 2 — Cross-referencing pass
        A second round of questions per suspect (up to ai_max_second_pass), now
        with all three initial interviews available for contradiction-probing.

    Phase 3 — Deduction
        The DeductionAgent analyses the complete transcript and reasons to the
        best (suspect_id, weapon, motive) accusation.

    Phase 4 — Evaluation
        The AccusationAgent scores and explains the result; session state is
        updated with the outcome and the page reruns to show the result screen.

    Duplicate-question guard:
        Before each question is sent, its Jaccard similarity is checked against
        all questions already asked to that suspect. If it exceeds the dedup
        threshold, one retry is attempted with an explicit reframe instruction
        appended to covered_topics. A second duplicate causes the turn to be skipped.
    """
    state = st.session_state.game.state

    st.markdown("""
    <div style="background: linear-gradient(90deg, transparent, #1a1a1a, transparent);
                padding: 10px; text-align: center;">
        <span style="font-family: 'Special Elite', cursive; font-size: 24px; color: #8B0000;">
            🤖 AI AGENT INVESTIGATION
        </span><br>
        <span style="font-family: 'Courier Prime', monospace; color: #666;">
            Automated interrogation in progress…
        </span>
    </div>
    """, unsafe_allow_html=True)

    remaining_turns = max(0, state.max_turns - state.total_turns)
    if remaining_turns == 0:
        st.warning("No turns remaining for the AI agent to interrogate.")
        st.session_state.ai_agent_active = False
        return

    progress_bar        = st.progress(0)
    status_text         = st.empty()
    think_text          = st.empty()
    questions_container = st.container()

    # Seed dedup tracking from any existing manual messages.
    asked_by_suspect: dict[str, list] = {
        sid: [
            normalize_to_set(m["content"])
            for m in st.session_state.messages.get(sid, [])
            if m.get("role") == "user" and m.get("content")
        ]
        for sid in SUSPECTS
    }

    # Pre-seed covered topics from any existing messages.
    covered_topics: dict[str, list] = {sid: [] for sid in SUSPECTS}
    for sid in SUSPECTS:
        for m in st.session_state.messages.get(sid, []):
            if m.get("role") == "user":
                covered_topics[sid].append(m["content"])

    questions_asked = 0
    total_budget    = remaining_turns
    cfg             = GAME_CONFIG

    # ------------------------------------------------------------------
    def _ask_one(sid: str) -> bool:
        """
        Plan and ask a single question to suspect `sid`.

        Implements the retry-on-duplicate logic:
          - First plan attempt: if duplicate, append an explicit 'avoid this'
            instruction to covered_topics and call the planner once more.
          - If the retry is also a duplicate, skip this turn.

        Returns True if a question was successfully asked, False otherwise.
        """
        nonlocal questions_asked

        if state.total_turns >= state.max_turns:
            return False

        suspect = SUSPECTS[sid]
        st.session_state.game.switch_suspect(sid)
        _update_suspect_sidebar(active_sid=sid, game_state=state)

        # First planning attempt.
        think_text.markdown(f"*🧠 Planning question for {suspect.name}…*")
        question = st.session_state.game.plan_next_question(
            sid, covered_topics[sid]
        )
        think_text.empty()

        q_set = normalize_to_set(question)

        # Retry once if the first attempt is a near-duplicate.
        if is_duplicate_question(q_set, asked_by_suspect[sid], threshold=cfg.ai_dedup_threshold):
            think_text.markdown(f"*🔄 Reframing question for {suspect.name}…*")
            reframe_topics = covered_topics[sid] + [
                f"[AVOID — too similar to previous] {question}"
            ]
            question = st.session_state.game.plan_next_question(sid, reframe_topics)
            think_text.empty()
            q_set = normalize_to_set(question)

            if is_duplicate_question(q_set, asked_by_suspect[sid], threshold=cfg.ai_dedup_threshold):
                return False  # Both attempts were duplicates — skip turn.

        # Ask the question.
        status_text.markdown(
            f"**🤖 Interrogating:** {suspect.name} — Question {questions_asked + 1}/{total_budget}"
        )
        response = st.session_state.game.interrogate(question)

        st.session_state.messages[sid].append({"role": "user",      "content": question})
        st.session_state.messages[sid].append({"role": "assistant", "content": response})

        asked_by_suspect[sid].append(q_set)
        covered_topics[sid].append(question)
        questions_asked += 1
        progress_bar.progress(min(questions_asked / max(1, total_budget), 1.0))

        # Live sidebar update.
        if st.session_state._sb_progress is not None:
            _render_sidebar_stats(
                state,
                st.session_state._sb_progress,
                st.session_state._sb_remaining,
                st.session_state._sb_interviewed,
                st.sidebar.empty(),
            )
        _update_suspect_sidebar(active_sid=sid, game_state=state)

        # Render exchange in the questions container.
        with questions_container:
            st.markdown(
                f"<div style='margin:4px 0;'>"
                f"<span style='color:#8B0000;font-family:Courier Prime,monospace;'>"
                f"🕵️ <b>Agent → {suspect.name}:</b></span><br>"
                f"<em>{question}</em></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin:4px 0 12px 0;'>"
                f"<span style='color:#aaa;font-family:Courier Prime,monospace;'>"
                f"🎭 <b>{suspect.name}:</b></span><br>"
                f"<em>{response}</em></div><hr style='border-color:#333;'>",
                unsafe_allow_html=True,
            )

        time.sleep(0.3)
        return True

    # ------------------------------------------------------------------
    # Phase 1 — Initial pass: question every suspect in order.
    # ------------------------------------------------------------------
    status_text.markdown("**🤖 Phase 1:** Initial interrogations…")
    for sid in ["s1", "s2", "s3"]:
        if state.total_turns >= state.max_turns:
            break
        per_suspect = 0
        while per_suspect < cfg.ai_max_first_pass and state.total_turns < state.max_turns:
            if _ask_one(sid):
                per_suspect += 1
            else:
                break

    # ------------------------------------------------------------------
    # Phase 2 — Cross-referencing: second pass with all intel available.
    # ------------------------------------------------------------------
    if state.total_turns < state.max_turns:
        status_text.markdown(
            "**🤖 Phase 2:** Cross-referencing and contradiction probing…"
        )
        for sid in ["s1", "s2", "s3"]:
            if state.total_turns >= state.max_turns:
                break
            per_suspect = 0
            while per_suspect < cfg.ai_max_second_pass and state.total_turns < state.max_turns:
                if _ask_one(sid):
                    per_suspect += 1
                else:
                    break

    # ------------------------------------------------------------------
    # Phase 3 — Deduction.
    # ------------------------------------------------------------------
    think_text.empty()
    status_text.markdown(
        "**🤖 Phase 3:** Reviewing all evidence and deducing the killer…"
    )
    with st.spinner("*The detective reviews the complete case file…*"):
        accused_id, weapon, motive = st.session_state.game.deduce_accusation()

    # Handle deduction failure gracefully.
    # deduce_accusation() returns (None, None, None) when the DeductionAgent's
    # output cannot be parsed or validated — see game_engine.py for detail.
    # Rather than silently guessing (which was the old behaviour), we surface
    # the error to the player and abort the AI agent run cleanly.
    if accused_id is None:
        logger.error("AI agent deduction failed — aborting Phase 4.")
        status_text.markdown(
            "**⚠️ AI Agent:** The detective could not reach a conclusion from "
            "the transcripts. Please try Manual Investigation or start a new case."
        )
        st.session_state.ai_agent_active = False
        return

    status_text.markdown(
        f"**🤖 AI Agent conclusion:** "
        f"{SUSPECTS[accused_id].name} · {weapon} · {motive}"
    )
    time.sleep(0.6)

    # ------------------------------------------------------------------
    # Phase 4 — Evaluation.
    # ------------------------------------------------------------------
    won, score, eval_text = st.session_state.game.make_accusation(
        accused_id, weapon, motive
    )

    st.session_state.game_over         = True
    st.session_state.accusation_result = {
        "won": won, "score": score, "eval_text": eval_text, "ai_agent": True,
    }
    st.session_state.ai_agent_active   = False

    progress_bar.progress(1.0)
    status_text.markdown("**✓ AI Agent:** Case analysis complete!")
    st.rerun()


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """
    Entry point — called by Streamlit on every render pass.

    Flow:
      1. Guard: require GROQ_API_KEY (prompt for it inline if missing).
      2. Initialise session state on first run.
      3. Render the page header.
      4. Render sidebar (mode selector, suspect list, progress stats).
      5. Render main panel (case briefing + appropriate content for current mode).
    """
    # --- API key guard ---
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY is not set!")
        st.markdown("""
        Set your key before launching:
        ```bash
        export GROQ_API_KEY="your-api-key"
        streamlit run app.py
        ```
        Or add it to a `.env` file in the project root.
        """)
        api_key = st.text_input("Enter your GROQ API key:", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("Key saved! Reloading…")
            st.rerun()
        return

    init_session_state()

    # --- Header ---
    st.markdown("""
    <h1 class='main-header'>🔍 AI MURDER MYSTERY</h1>
    <h3 class='sub-header'>The Blackwood Mansion Affair</h3>
    <p style="text-align: center; color: #444; font-family: 'Courier Prime', monospace; font-size: 12px;">
        A game of deception, deduction, and dark secrets
    </p>
    """, unsafe_allow_html=True)

    # --- Sidebar: Mode selector ---
    st.sidebar.markdown(
        '<div class="sidebar-header">🎮 INVESTIGATION MODE</div>',
        unsafe_allow_html=True,
    )
    mode_col1, mode_col2 = st.sidebar.columns(2)
    with mode_col1:
        if st.button(
            "👁️ Manual\nInvestigate",
            key="mode_manual",
            use_container_width=True,
            disabled=st.session_state.ai_agent_active,
        ):
            st.session_state.game_mode = "manual"
            st.rerun()
    with mode_col2:
        if st.button(
            "🤖 AI Agent\nSolve",
            key="mode_agent",
            use_container_width=True,
            disabled=st.session_state.game_over,
        ):
            st.session_state.game_mode      = "ai_agent"
            st.session_state.ai_agent_active = True
            st.rerun()

    mode_icon = "👁️" if st.session_state.game_mode == "manual" else "🤖"
    mode_text = (
        "Manual Investigation"
        if st.session_state.game_mode == "manual"
        else "AI Agent Mode"
    )
    st.sidebar.markdown(f"**Current Mode:** {mode_icon} {mode_text}")
    st.sidebar.markdown("---")

    render_suspect_selector()
    render_game_status()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 NEW CASE", use_container_width=True):
        reset_game()
        st.rerun()

    # --- Main panel ---
    render_case_briefing()

    if st.session_state.ai_agent_active:
        run_ai_agent_interrogation()
    elif st.session_state.game_over and st.session_state.accusation_result:
        render_game_result()
    elif st.session_state.game_mode == "manual":
        render_chat_interface()
        render_notes_section()
        render_accusation_form()
    else:
        st.info("🤖 Click 'AI Agent Solve' to begin automated investigation.")


if __name__ == "__main__":
    main()
