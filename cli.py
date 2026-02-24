"""
cli.py
======
Command-line interface for AI Murder Mystery: The Blackwood Mansion.

Provides a text-based game loop for development, testing, and running the
game without Streamlit. All game logic is delegated to MurderMysteryGame;
this module only handles I/O.

Usage:
    python cli.py

Commands during play:
    /suspect <id>                — switch to a different suspect (e.g. /suspect s2)
    /suspects                    — list all suspect IDs and names
    /status                      — show turn count and interrogated suspects
    /accuse <id> <weapon> <motive> — make a final accusation
    /quit                        — exit the game
"""

from __future__ import annotations

import logging
import os

from case_data import CASE_FILE, SUSPECTS, VALID_MOTIVES, VALID_WEAPONS
from game_engine import MurderMysteryGame


def run_cli() -> None:
    """
    Main CLI game loop.

    Validates the GROQ_API_KEY environment variable, initialises the game engine,
    prints the case briefing, then processes player input in a loop until the
    player makes an accusation or quits.
    """
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("  export GROQ_API_KEY='your-key-here'")
        return

    game = MurderMysteryGame()
    v    = CASE_FILE["victim"]

    # --- Case briefing banner ---
    print("\n" + "=" * 60)
    print("   AI MURDER MYSTERY: THE BLACKWOOD MANSION")
    print("=" * 60)
    print(f"\nVICTIM     : {v['name']}")
    print(f"TIME       : {v['time_of_death']}")
    print(f"LOCATION   : {v['location']}")
    print(f"CAUSE      : {v['cause']}")
    print("\nCommands: /suspect <id>, /suspects, /accuse <id> <weapon> <motive>, /status, /quit")
    print("-" * 60)

    while True:
        suspect   = game.get_current_suspect()
        user_input = input(
            f"\n[{game.state.total_turns + 1}/{game.state.max_turns}] "
            f"[You → {suspect.name}]: "
        ).strip()

        if not user_input:
            continue

        lower = user_input.lower()

        # ---- Command: quit ----
        if lower in {"/quit", "quit", "exit"}:
            print("Thanks for playing!")
            break

        # ---- Command: list suspects ----
        if lower == "/suspects":
            for sid, s in SUSPECTS.items():
                print(f"  {sid} – {s.name} ({s.role})")
            continue

        # ---- Command: status ----
        if lower == "/status":
            st = game.state
            print(f"  Turns used   : {st.total_turns}/{st.max_turns}")
            print(f"  Interviewed  : {sorted(st.suspects_interviewed)}")
            continue

        # ---- Command: switch suspect ----
        if lower.startswith("/suspect "):
            parts = lower.split()
            if len(parts) < 2:
                print("Usage: /suspect <id>")
                continue
            new_id = parts[1]
            if game.switch_suspect(new_id):
                print(f"Now interrogating: {SUSPECTS[new_id].name}")
            else:
                print(
                    f"Unknown suspect ID: {new_id}. "
                    f"Valid IDs: {list(SUSPECTS.keys())}"
                )
            continue

        # ---- Command: accuse (show options) ----
        if lower == "/accuse":
            print("Usage: /accuse <suspect_id> <weapon> <motive>")
            print(f"Suspects : {list(SUSPECTS.keys())}")
            print(f"Weapons  : {VALID_WEAPONS}")
            print(f"Motives  : {VALID_MOTIVES}")
            continue

        # ---- Command: accuse (submit) ----
        if lower.startswith("/accuse "):
            parts = user_input[8:].split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: /accuse <suspect_id> <weapon> <motive>")
                continue
            accused_id, weapon, motive = parts
            won, score, text = game.make_accusation(accused_id, weapon, motive)
            print(text)
            print(
                f"\n{'🎉 CASE SOLVED!' if won else '❌ CASE UNSOLVED...'} "
                f"Score: {score}/100"
            )
            break

        # ---- Normal interrogation ----
        if game.state.total_turns >= game.state.max_turns:
            print("No turns left! Use /accuse to make your accusation.")
            continue

        response = game.interrogate(user_input)
        print(f"\n[{suspect.name}]: {response}")


if __name__ == "__main__":
    # Configure logging at the entry point so all murder_mystery.* loggers
    # emit to stdout at INFO level. Swap StreamHandler for a FileHandler here
    # to redirect logs to disk without touching any other module.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_cli()
