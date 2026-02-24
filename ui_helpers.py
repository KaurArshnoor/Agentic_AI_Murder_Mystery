"""
ui_helpers.py
=============
Stateless UI utility functions for the Streamlit interface.

These functions produce or manipulate UI elements but carry no game state
of their own — they receive all required data as arguments. Keeping them
separate from app.py means they can be imported and tested in isolation
without a live Streamlit session.

Contains:
  - normalize_to_set()    : text → word set (for dedup comparisons)
  - jaccard()             : Jaccard similarity between two word sets
  - is_duplicate_question(): similarity gate for the AI planner
  - build_css()           : returns the full dark-noir CSS string
"""

from __future__ import annotations

import string
from typing import List, Set


# ---------------------------------------------------------------------------
# Question deduplication helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "the", "a", "an", "at", "in", "on", "to", "of", "and", "is", "was",
    "were", "i", "you", "he", "she", "it", "they", "we", "my", "your",
    "his", "her", "their", "that", "this", "do", "did", "have", "has",
}
"""
Common stop words excluded from Jaccard similarity calculations.

Removing these high-frequency words means similarity scores reflect the
meaningful content of questions rather than grammatical scaffolding.
"""


def normalize_to_set(text: str) -> Set[str]:
    """
    Convert a question string to a set of meaningful, lowercase words.

    Steps:
      1. Lowercase.
      2. Strip punctuation (replace with spaces).
      3. Split on whitespace.
      4. Remove stop words and single-character tokens.

    Args:
        text: Any string, typically a question from the interrogation planner.

    Returns:
        Set of meaningful word tokens.

    Example:
        >>> normalize_to_set("Where were you at the time of the murder?")
        {'where', 'time', 'murder'}
    """
    txt = (text or "").lower()
    txt = txt.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    return {
        w for w in txt.split()
        if w and w not in _STOP_WORDS and len(w) > 2
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Compute the Jaccard similarity between two word sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns 1.0 when both sets are empty (treat as identical),
    and 0.0 when one is empty and the other is not.

    Args:
        a, b: Word-token sets produced by normalize_to_set().

    Returns:
        Float in [0.0, 1.0].
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_duplicate_question(
    q_set: Set[str],
    seen: List[Set[str]],
    threshold: float = 0.7,
) -> bool:
    """
    Return True if `q_set` is too similar to any previously-asked question.

    Used by the AI agent to detect when the InterrogationPlanner has generated
    a question that covers the same ground as an earlier one, triggering a
    retry with explicit reframe instructions.

    Args:
        q_set:     Word-set for the candidate question (from normalize_to_set).
        seen:      List of word-sets for all questions already asked to this suspect.
        threshold: Jaccard score at or above which the question is considered a duplicate.
                   Default 0.7; the AI agent caller uses 0.65 for a tighter gate.

    Returns:
        True if the question is a near-duplicate of a previously asked one.
    """
    return any(jaccard(q_set, prev) >= threshold for prev in seen)


# ---------------------------------------------------------------------------
# Dark-noir CSS
# ---------------------------------------------------------------------------

def build_css() -> str:
    """
    Return the full dark-noir CSS string injected into the Streamlit app.

    Factored out of app.py so the main file reads clearly and the CSS
    can be edited or themed without scrolling through UI logic.

    Returns:
        A raw CSS string (without <style> tags — the caller wraps it).
    """
    return """
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Courier+Prime:wght@400;700&display=swap');

    /* ── Global dark background ── */
    html, body, .stApp, .reportview-container, .appview-container,
    .main, .block-container, .block-container > div {
        background: linear-gradient(180deg, #0a0a0a 0%, #141414 60%, #0d0d0d 100%) !important;
        color: #c0c0c0 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: #0d0d0d !important;
        background-color: #0d0d0d !important;
        border-right: 1px solid #222 !important;
    }
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background: #0d0d0d !important;
        background-color: #0d0d0d !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #c0c0c0 !important; }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        background: #0d0d0d !important;
        color: #8B0000 !important;
    }

    /* ── Typography ── */
    .main-header {
        text-align: center; color: #8B0000;
        font-family: 'Special Elite', cursive;
        text-shadow: 2px 2px 4px #000; letter-spacing: 3px;
    }
    .sub-header {
        text-align: center; color: #666;
        font-family: 'Courier Prime', monospace; font-style: italic;
    }

    /* ── Cards & panels ── */
    .case-file {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 25px; border-radius: 5px;
        border-left: 4px solid #8B0000; border-top: 1px solid #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        font-family: 'Courier Prime', monospace;
    }
    .case-file h3 { color: #8B0000; font-family: 'Special Elite', cursive; letter-spacing: 2px; }
    .suspect-card {
        background: linear-gradient(145deg, #1a1a1a, #252525);
        padding: 15px; border-radius: 8px; margin: 10px 0;
        border: 1px solid #333; transition: all 0.3s ease;
    }
    .suspect-card:hover { border-color: #8B0000; box-shadow: 0 0 10px rgba(139,0,0,0.3); }

    /* ── Chat messages ── */
    .stChatMessage {
        background-color: #1a1a1a !important;
        border: 2px solid #8B0000; border-radius: 10px; padding: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        font-family: 'Courier Prime', monospace;
    }
    .typewriter-text { font-family: 'Special Elite', cursive; color: #c0c0c0; line-height: 1.8; }

    /* ── Notes section ── */
    .detective-notes {
        background: linear-gradient(145deg, #2a2a1a, #1a1a0a);
        padding: 20px; border-radius: 5px; border: 1px solid #4a4a2a;
        font-family: 'Courier Prime', monospace; min-height: 150px;
    }
    .notes-header {
        color: #8B0000; font-family: 'Special Elite', cursive;
        border-bottom: 1px solid #4a4a2a; padding-bottom: 10px; margin-bottom: 15px;
    }

    /* ── Score display ── */
    .score-display {
        font-size: 64px; font-weight: bold; text-align: center;
        color: #8B0000; font-family: 'Special Elite', cursive;
        text-shadow: 2px 2px 4px #000;
    }

    /* ── Sidebar header ── */
    .sidebar-header {
        color: #8B0000; font-family: 'Special Elite', cursive;
        letter-spacing: 2px; text-align: center; padding: 10px;
        border-bottom: 1px solid #333;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
        color: #c0c0c0; border: 1px solid #444;
        font-family: 'Courier Prime', monospace; transition: all 0.3s ease;
        min-height: 60px !important; height: auto !important;
        white-space: normal !important; overflow-wrap: anywhere !important;
        display: flex !important; align-items: center !important;
        justify-content: center !important; text-align: center !important;
        padding: 10px !important;
    }
    .stButton > button:hover { border-color: #8B0000; color: #8B0000; box-shadow: 0 0 10px rgba(139,0,0,0.3); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(145deg, #8B0000, #5a0000); color: #fff; border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(145deg, #a00000, #6a0000); box-shadow: 0 0 15px rgba(139,0,0,0.5);
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader { background-color: #1a1a1a; color: #8B0000; font-family: 'Special Elite', cursive; }

    /* ── Progress bars ── */
    .stProgress > div > div,
    [data-testid="stSidebar"] .stProgress > div > div { background-color: #8B0000 !important; }
    .stProgress > div,
    [data-testid="stSidebar"] .stProgress > div {
        background-color: #2a2a2a !important; border-radius: 4px;
    }

    /* ── Inputs ── */
    .stTextArea textarea, .stTextInput input, .stChatInput textarea, .stChatInput input {
        background-color: #141414 !important; color: #c0c0c0 !important;
        border: 1px solid #333 !important; border-radius: 8px !important;
        font-family: 'Courier Prime', monospace;
    }

    /* ── Bottom bar & footer ── */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
        background-color: #0a0a0a !important; border-top: 1px solid #333 !important;
    }
    footer, .stFooter { background: #0a0a0a !important; color: #c0c0c0 !important; }

    /* ── Vignette overlay ── */
    .vignette {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.6) 100%);
        z-index: 0;
    }
"""
