"""
Train AXIOM on a single Gameworld game.

Usage:
    python experiments/run_experiment.py --game bounce --steps 10000
    python experiments/run_experiment.py --game bounce --config experiments/configs/bounce.yaml
"""

import argparse
from pathlib import Path

import yaml


def load_config(game: str, config_path: str | None = None) -> dict:
    """Load default config, then overlay game-specific and CLI overrides."""
    config_dir = Path(__file__).parent / "configs"

    with open(config_dir / "default.yaml") as f:
        config = yaml.safe_load(f)

    game_config_path = config_dir / f"{game}.yaml"
    if game_config_path.exists():
        with open(game_config_path) as f:
            game_overrides = yaml.safe_load(f) or {}
        config.update(game_overrides)

    if config_path:
        with open(config_path) as f:
            cli_overrides = yaml.safe_load(f) or {}
        config.update(cli_overrides)

    config["game"] = game
    return config


def train(config: dict):
    """Main training loop."""
    game = config["game"]
    steps = config["steps"]
    seed = config["seed"]

    print(f"Game: {game} | Steps: {steps} | Seed: {seed}")
    print(f"Config: {config}")

    # TODO: Initialize environment
    # TODO: Initialize AXIOM agent
    # TODO: Training loop: observe -> act -> step -> repeat
    # TODO: Log metrics and save results

    raise NotImplementedError("Training loop not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Train AXIOM on a Gameworld game")
    parser.add_argument("--game", type=str, default="bounce")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.game, args.config)
    if args.steps is not None:
        config["steps"] = args.steps
    if args.seed is not None:
        config["seed"] = args.seed

    train(config)


if __name__ == "__main__":
    main()
