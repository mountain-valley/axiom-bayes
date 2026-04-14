"""Command-line interface for AXIOM."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="AXIOM: Active eXpanding Inference with Object-centric Models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train AXIOM on a game")
    train_parser.add_argument("--game", type=str, default="bounce", help="Game name")
    train_parser.add_argument("--steps", type=int, default=10_000, help="Number of interaction steps")
    train_parser.add_argument("--seed", type=int, default=0, help="Random seed")
    train_parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config (overrides defaults)"
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--game", type=str, required=True)
    eval_parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)


def _train(args):
    print(f"Training AXIOM on {args.game} for {args.steps} steps (seed={args.seed})")
    raise NotImplementedError("Training loop not yet implemented")


def _evaluate(args):
    print(f"Evaluating checkpoint {args.checkpoint} on {args.game}")
    raise NotImplementedError("Evaluation not yet implemented")


if __name__ == "__main__":
    main()
