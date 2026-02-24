"""
config.py
=========
Central configuration module for AI Murder Mystery: The Blackwood Mansion.

All tunable constants, model identifiers, scoring weights, and game-balance
parameters live here so they can be adjusted without touching business logic.

Usage:
    from config import ModelConfig, ScoringConfig, GameConfig, CRITIQUE_TRIGGERS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Groq model identifiers used across the system.

    Attributes:
        suspect_model:  Large, high-quality model for nuanced in-character roleplay.
        utility_model:  Smaller, faster model for filter / scoring / judging tasks.
                        Using a cheaper model here keeps latency and cost down for
                        tasks that do not require deep creative reasoning.
    """
    suspect_model: str = "llama-3.3-70b-versatile"
    utility_model: str = "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringConfig:
    """
    Weights and thresholds for the deterministic scoring function.

    Points breakdown (max 100):
        suspect_points + weapon_points + motive_points  ≤ 100

    Efficiency bonuses / penalties are applied on top and the total is
    clamped to [0, 100].

    Attributes:
        suspect_points:     Points awarded for identifying the correct killer.
        weapon_points:      Points awarded for the correct murder weapon.
        motive_points:      Points awarded for the correct motive.
        efficiency_fast:    Bonus turns threshold — solved in ≤ this many turns.
        efficiency_medium:  Medium bonus turns threshold.
        efficiency_bonus_fast:    Bonus points for the fast threshold.
        efficiency_bonus_medium:  Bonus points for the medium threshold.
        efficiency_penalty_start: Turns beyond which penalties accumulate.
        efficiency_penalty_per_turn: Penalty points per extra turn.
        efficiency_penalty_cap:  Maximum total penalty applied.
    """
    suspect_points: int = 40
    weapon_points:  int = 30
    motive_points:  int = 30

    efficiency_fast:   int = 10
    efficiency_medium: int = 15

    efficiency_bonus_fast:   int = 10
    efficiency_bonus_medium: int = 5

    efficiency_penalty_start:    int = 25
    efficiency_penalty_per_turn: int = 2
    efficiency_penalty_cap:      int = 20


# ---------------------------------------------------------------------------
# Game-balance parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameConfig:
    """
    Top-level game-balance and AI-agent behaviour settings.

    Attributes:
        max_turns:              Hard cap on total interrogation turns per session.
        context_compress_after: Turn number after which older exchanges are
                                summarised to prevent context-window overflow.
        history_verbatim_recent: Number of most-recent exchanges kept verbatim
                                when context compression is active.
        history_verbatim_early:  Number of exchanges shown verbatim before
                                compression kicks in (early-game).
        planner_history_window:  Number of prior exchanges fed to the Planner
                                per suspect (for context without overflowing).
        planner_other_window:    Number of recent exchanges from OTHER suspects
                                fed to the Planner (for cross-referencing).
        ai_max_first_pass:       Questions per suspect in AI-agent Phase 1.
        ai_max_second_pass:      Questions per suspect in AI-agent Phase 2.
        ai_dedup_threshold:      Jaccard similarity threshold above which a
                                planner question is considered a duplicate.
    """
    max_turns:               int = 30
    context_compress_after:  int = 20
    history_verbatim_recent: int = 4
    history_verbatim_early:  int = 6

    planner_history_window:  int = 8
    planner_other_window:    int = 4

    ai_max_first_pass:  int = 4
    ai_max_second_pass: int = 2
    ai_dedup_threshold: float = 0.65


# ---------------------------------------------------------------------------
# Singleton instances (import-ready)
# ---------------------------------------------------------------------------

MODEL_CONFIG   = ModelConfig()
SCORING_CONFIG = ScoringConfig()
GAME_CONFIG    = GameConfig()


# ---------------------------------------------------------------------------
# Critique trigger keywords
# ---------------------------------------------------------------------------

CRITIQUE_TRIGGERS: FrozenSet[str] = frozenset({
    # Weapon hints
    "candlestick", "candleholder", "brass", "fireplace", "mantelpiece",
    "poker", "letter opener", "rope", "poison", "weapon", "object",
    # Culprit hints
    "lydia", "killer", "murderer", "culprit", "did it", "guilty",
    # Confession fragments
    "i killed", "i hit", "i struck", "i was in the library", "i murdered",
    # Accomplice hints
    "hid", "hidden", "covered up", "alibi", "coached", "fake timeline",
    # Timeline hints
    "23:15", "23:20", "11:15", "11:20",
})
"""
Keywords that trigger the CritiqueAgent on a suspect's raw response.

The CritiqueAgent is an 8b-model call that checks for accidental leaks of
the true killer, weapon, or timeline. Running it on every turn would add
unnecessary latency; instead we only invoke it when the raw response contains
at least one of these sentinel words.

Approximately 50 % of turns pass through without critique, saving ~300 ms each.
"""
