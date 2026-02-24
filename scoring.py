"""
scoring.py
==========
Deterministic, side-effect-free scoring logic.

Extracted from the game engine so it can be unit-tested independently and
adjusted by changing ScoringConfig values in config.py without touching
any game logic or UI code.
"""

from __future__ import annotations

from config import SCORING_CONFIG


def calculate_score(
    correct_suspect: bool,
    correct_weapon:  bool,
    correct_motive:  bool,
    total_turns:     int,
) -> int:
    """
    Compute the player's final score in the range [0, 100].

    Points breakdown:
        correct_suspect → +suspect_points  (default: 40)
        correct_weapon  → +weapon_points   (default: 30)
        correct_motive  → +motive_points   (default: 30)
        Subtotal max    → 100

    Efficiency modifier (applied after the component subtotal):
        ≤ efficiency_fast   turns  → +efficiency_bonus_fast   (default: +10)
        ≤ efficiency_medium turns  → +efficiency_bonus_medium (default: +5)
        > efficiency_penalty_start → -efficiency_penalty_per_turn per extra turn,
                                     capped at -efficiency_penalty_cap (default: -20)

    Final score is clamped to [0, 100].

    Args:
        correct_suspect: True if the player named the right killer.
        correct_weapon:  True if the player named the right weapon.
        correct_motive:  True if the player named the right motive.
        total_turns:     Total questions asked across all suspects this session.

    Returns:
        Integer score in [0, 100].

    Examples:
        >>> calculate_score(True, True, True, 8)
        110  → clamped to 100
        >>> calculate_score(True, False, False, 20)
        40
        >>> calculate_score(False, False, False, 5)
        0
    """
    cfg = SCORING_CONFIG

    # --- Component score ---
    base = 0
    if correct_suspect:
        base += cfg.suspect_points
    if correct_weapon:
        base += cfg.weapon_points
    if correct_motive:
        base += cfg.motive_points

    # --- Efficiency modifier ---
    efficiency = 0
    if total_turns <= cfg.efficiency_fast:
        efficiency = cfg.efficiency_bonus_fast
    elif total_turns <= cfg.efficiency_medium:
        efficiency = cfg.efficiency_bonus_medium
    elif total_turns > cfg.efficiency_penalty_start:
        raw_penalty = (total_turns - cfg.efficiency_penalty_start) * cfg.efficiency_penalty_per_turn
        efficiency  = -min(cfg.efficiency_penalty_cap, raw_penalty)

    return max(0, min(100, base + efficiency))
