"""
models.py
=========
Shared data models for AI Murder Mystery: The Blackwood Mansion.

Contains:
  - DeductionResult   : Pydantic schema for the DeductionAgent's structured output.
  - SuspectProfile    : Dataclass describing a suspect's public / private persona.
  - GameState         : Mutable dataclass tracking per-session player progress.

Keeping these in one module guarantees a single source of truth for data
shapes used across agents.py, game_engine.py, and the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Set


# ---------------------------------------------------------------------------
# Pydantic structured output schema
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class DeductionResult(BaseModel):
    """
    Validated output schema for DeductionAgent.

    Using a Pydantic model instead of regex-based string parsing guarantees
    that the accusation fields are always one of the enumerated valid values,
    even when the LLM formats its JSON differently across runs.

    Fields:
        suspect_id: One of the three suspect IDs defined in case_data.py.
        weapon:     Exact weapon string from the VALID_WEAPONS list.
        motive:     Exact motive string from the VALID_MOTIVES list.
        reasoning:  The agent's step-by-step chain-of-thought. This is logged
                    for debugging but is never shown to the player.
    """

    suspect_id: Literal["s1", "s2", "s3"]
    weapon: Literal[
        "brass candlestick",
        "letter opener",
        "fireplace poker",
        "poison",
        "rope",
    ]
    motive: Literal[
        "inheritance",
        "jealousy",
        "revenge",
        "blackmail",
        "affair",
    ]
    reasoning: str


# ---------------------------------------------------------------------------
# Suspect profile
# ---------------------------------------------------------------------------

@dataclass
class SuspectProfile:
    """
    Full description of a suspect used to build their SuspectAgent system prompt.

    Attributes:
        id:            Short identifier (e.g. "s1") used as a dict key everywhere.
        name:          Display name shown in the UI and spoken in prompts.
        persona:       High-level character description that shapes how the LLM
                       responds — their tone, mannerisms, and emotional state.
        public_info:   The alibi / cover story the suspect will openly repeat.
        secret_info:   The hidden truth that must NEVER be revealed verbatim.
        role:          "killer" | "accomplice" | "innocent" — private metadata
                       used by agents and the accusation judge; never shown to players.
        hard_redlines: Strings that must never appear in the suspect's output.
                       The CritiqueAgent enforces these as a secondary safety net.
    """

    id:             str
    name:           str
    persona:        str
    public_info:    str
    secret_info:    str
    role:           str
    hard_redlines:  List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Mutable snapshot of everything that changes as the player interrogates suspects.

    This object is owned by MurderMysteryGame and mutated in place as turns
    progress.  The Streamlit app reads it (read-only) for rendering the sidebar
    progress indicators.

    Attributes:
        total_turns:         Total questions asked across all suspects.
        turns_per_suspect:   Per-suspect question counts (keyed by suspect ID).
        suspects_interviewed: Set of suspect IDs the player has spoken to at
                              least once this session.
        accusation_made:     True once make_accusation() has been called.
        game_won:            True if the player correctly identified the culprit.
        final_score:         Numeric score in [0, 100] from _calculate_score().
        max_turns:           Hard cap; sourced from GameConfig.max_turns.
    """

    total_turns:          int  = 0
    turns_per_suspect:    Dict[str, int] = field(default_factory=dict)
    suspects_interviewed: Set[str]       = field(default_factory=set)
    accusation_made:      bool = False
    game_won:             bool = False
    final_score:          int  = 0
    max_turns:            int  = 30  # overwritten by MurderMysteryGame.__init__

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_turn(self, suspect_id: str) -> None:
        """
        Record that one question was asked to `suspect_id`.

        Increments the global turn counter, updates the per-suspect counter,
        and marks the suspect as interviewed for progress tracking.
        """
        self.total_turns += 1
        self.turns_per_suspect[suspect_id] = (
            self.turns_per_suspect.get(suspect_id, 0) + 1
        )
        self.suspects_interviewed.add(suspect_id)

    def reset(self) -> None:
        """Reset all mutable fields to their initial values for a new game."""
        self.total_turns          = 0
        self.turns_per_suspect    = {}
        self.suspects_interviewed = set()
        self.accusation_made      = False
        self.game_won             = False
        self.final_score          = 0
