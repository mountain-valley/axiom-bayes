"""
Run AXIOM across all 10 Gameworld games with multiple seeds.

Usage:
    python experiments/sweep.py --seeds 10 --steps 10000
"""

import argparse

from envs.gameworld import GAME_NAMES
from experiments.run_experiment import load_config, train


def main():
    parser = argparse.ArgumentParser(description="Sweep all games × seeds")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--games", nargs="+", default=None, help="Subset of games (default: all)")
    args = parser.parse_args()

    games = args.games or GAME_NAMES

    for game in games:
        for seed in range(args.seeds):
            config = load_config(game)
            config["steps"] = args.steps
            config["seed"] = seed
            print(f"\n=== Game: {game} | Seed: {seed} ===")
            try:
                train(config)
            except NotImplementedError:
                print("  (skipped — not yet implemented)")


if __name__ == "__main__":
    main()
