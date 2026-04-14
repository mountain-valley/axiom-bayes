"""
Run AXIOM ablation experiments.

Ablation variants from the paper:
- no BMR: disable Bayesian Model Reduction
- no IG: disable information gain term in planning
- fixed_distance: use fixed interaction distance instead of learned

Usage:
    python experiments/run_ablation.py --steps 10000
"""

import argparse

from experiments.run_experiment import load_config, train


ABLATION_VARIANTS = {
    "no_bmr": {"bmr_interval": 0},
    "no_ig": {"info_gain_weight": 0.0},
    "fixed_distance": {"fixed_interaction_distance": True},
}


def main():
    parser = argparse.ArgumentParser(description="Run AXIOM ablation experiments")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ABLATION_VARIANTS.keys()),
        choices=list(ABLATION_VARIANTS.keys()),
    )
    parser.add_argument("--games", nargs="+", default=None, help="Subset of games (default: all)")
    args = parser.parse_args()

    from envs.gameworld import GAME_NAMES

    games = args.games or GAME_NAMES

    for variant_name in args.variants:
        overrides = ABLATION_VARIANTS[variant_name]
        for game in games:
            config = load_config(game)
            config["steps"] = args.steps
            config.update(overrides)
            config["ablation"] = variant_name
            print(f"\n=== Ablation: {variant_name} | Game: {game} ===")
            try:
                train(config)
            except NotImplementedError:
                print("  (skipped — not yet implemented)")


if __name__ == "__main__":
    main()
