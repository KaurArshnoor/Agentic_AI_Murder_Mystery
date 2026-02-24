"""
case_data.py
============
All narrative content for the Blackwood Mansion murder case.

Centralising story data here means you can swap out the entire mystery
(suspects, victim, truth, weapon/motive lists) without touching any
agent, engine, or UI logic.

To create a new case:
    1. Replace the constants below with your new story.
    2. Update DeductionResult's Literal types in models.py to match
       your VALID_WEAPONS and VALID_MOTIVES.
    3. Keep the dict / dataclass shapes identical so nothing else breaks.
"""

from __future__ import annotations

from typing import Dict, List

from models import SuspectProfile


# ---------------------------------------------------------------------------
# Valid accusation options
# ---------------------------------------------------------------------------

VALID_WEAPONS: List[str] = [
    "brass candlestick",
    "letter opener",
    "fireplace poker",
    "poison",
    "rope",
]
"""
Complete list of possible murder weapons.
Must stay in sync with DeductionResult.weapon Literal in models.py.
Shown to players in the accusation form and validated by the judge.
"""

VALID_MOTIVES: List[str] = [
    "inheritance",
    "jealousy",
    "revenge",
    "blackmail",
    "affair",
]
"""
Complete list of possible murder motives.
Must stay in sync with DeductionResult.motive Literal in models.py.
"""


# ---------------------------------------------------------------------------
# Case file (the ground truth the judge uses to score accusations)
# ---------------------------------------------------------------------------

CASE_FILE: Dict = {
    "case_id": "mansion_murder_01",

    # Public victim information shown on the case briefing card.
    "victim": {
        "name":          "Victor Hale",
        "time_of_death": "23:15",
        "location":      "Library of Blackwood Mansion",
        "cause":         "Blunt force trauma to the head",
    },

    # Ground truth used only by the AccusationAgent (judge). Never revealed to players
    # directly — the LLM judge receives it alongside the player's accusation.
    "truth": {
        "culprit_id": "s1",
        "method":     "brass candlestick",
        "motive":     "inheritance",
        "timeline": [
            "22:30 - Victor seen arguing with Lydia (s1) in the study.",
            "23:00 - Neighbours heard shouting.",
            "23:15 - Victor killed in the library.",
            "23:20 - Eleanor (s3) glimpsed Lydia rushing away from the library.",
        ],
    },

    # Case-level strings that must NEVER appear verbatim in any suspect response.
    # The CritiqueAgent is given these alongside the suspect's personal hard_redlines.
    "redlines": [
        "Lydia killed Victor",
        "Lydia is the killer",
        "The culprit is Lydia Blackwood",
        "He was killed with a brass candlestick",
    ],
}


# ---------------------------------------------------------------------------
# Suspect profiles
# ---------------------------------------------------------------------------

SUSPECTS: Dict[str, SuspectProfile] = {

    # ------------------------------------------------------------------
    # s1 — The killer
    # ------------------------------------------------------------------
    "s1": SuspectProfile(
        id="s1",
        name="Lydia Blackwood",
        role="killer",
        persona=(
            "You are Lydia Blackwood, an elegant, composed heiress. "
            "You are evasive, controlled, and you dislike being challenged."
        ),
        public_info=(
            "You claim you were in your bedroom reading a novel "
            "when the murder occurred."
        ),
        secret_info=(
            "You killed Victor in the library at 23:15 using a brass candlestick "
            "after a heated argument about the inheritance."
        ),
        hard_redlines=[
            "I killed Victor",
            "I murdered Victor",
            "I am the killer",
            "I hit Victor with the candlestick",
            "I was in the library at 23:15",
        ],
    ),

    # ------------------------------------------------------------------
    # s2 — The accomplice
    # ------------------------------------------------------------------
    "s2": SuspectProfile(
        id="s2",
        name="Dr. Marcus Vale",
        role="accomplice",
        persona=(
            "You are Dr. Marcus Vale, the charming family doctor with a dry "
            "sense of humour. You deflect uncomfortable questions with sarcasm "
            "or intellectual talk."
        ),
        public_info=(
            "You say you left the mansion at 22:45 after a routine check-up."
        ),
        secret_info=(
            "You helped Lydia hide the brass candlestick in the fireplace "
            "and coached her on a false alibi timeline."
        ),
        hard_redlines=[
            "I helped Lydia after the murder",
            "I hid the candlestick",
            "I helped hide the weapon",
        ],
    ),

    # ------------------------------------------------------------------
    # s3 — The innocent witness
    # ------------------------------------------------------------------
    "s3": SuspectProfile(
        id="s3",
        name="Eleanor Wright",
        role="innocent",
        persona=(
            "You are Eleanor Wright, the anxious but loyal housekeeper. "
            "You speak softly, are easily flustered, and worry about losing "
            "your job."
        ),
        public_info=(
            "You say you were in the basement doing laundry during the incident."
        ),
        secret_info=(
            "You saw Lydia rushing down from the library around 23:20, "
            "but you are too scared to say it plainly."
        ),
        hard_redlines=[
            "I saw Lydia kill Victor",
            "I know for sure Lydia is the killer",
            "Lydia was definitely in the library at 23:15",
        ],
    ),
}
"""
Dict mapping suspect ID → SuspectProfile.

The ordering (s1, s2, s3) is used in several places:
    - The AI agent iterates suspects in this order for Phase 1 and Phase 2.
    - The accusation form renders options in this order.
Keep IDs stable between sessions because they are stored in session state.
"""
