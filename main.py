"""Root CLI dispatcher for multi-BS ISAC training pipelines."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ppo.train_hierarchical import main as train_hierarchical_main
from ppo.train_ppo import main as train_ppo_main


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse root dispatcher flags and preserve trainer-specific arguments."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Multi-BS ISAC Training Dispatcher",
        add_help=False,
    )
    parser.add_argument(
        "--algo",
        choices=("ppo", "hierarchical"),
        default="ppo",
        help="Training pipeline to run (default: ppo)",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show help for dispatcher or selected --algo",
    )
    args, remaining = parser.parse_known_args(raw_argv)
    if args.help:
        has_explicit_algo = any(
            token == "--algo" or token.startswith("--algo=")
            for token in raw_argv
        )
        if has_explicit_algo:
            remaining = [*remaining, "--help"]
        else:
            parser.print_help()
            print(
                "\nExamples:\n"
                "  python main.py --algo ppo --help\n"
                "  python main.py --algo hierarchical --help"
            )
            raise SystemExit(0)
    return args, remaining


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch execution to train_ppo or train_hierarchical."""
    args, remaining = parse_args(argv)
    if args.algo == "hierarchical":
        train_hierarchical_main(remaining)
    else:
        train_ppo_main(remaining)


if __name__ == "__main__":
    main()

"""
===============================================================================
USAGE GUIDE / CLI CHEATSHEET (main.py):
===============================================================================

1. TRAIN MAPPO (Default algorithm):
   python main.py
   python main.py --algo ppo
   python main.py --algo ppo --single-bwp
   python main.py --algo ppo --resume results/checkpoints/mappo/episode_000100.pt
   python main.py --algo ppo --figures-only

2. TRAIN HIERARCHICAL ISAC (SAC at Frame level + DQN at Subframe level):
   python main.py --algo hierarchical
   python main.py --algo hierarchical --episodes 500
   python main.py --algo hierarchical --single-bwp
   python main.py --algo hierarchical --subframes-per-frame 10 --episodes 200
   python main.py --algo hierarchical --resume results/checkpoints/hierarchical/episode_000050.pt
   python main.py --algo hierarchical --figures-only

3. VIEW HELP & CLI OPTIONS:
   python main.py --help
   python main.py --algo ppo --help
   python main.py --algo hierarchical --help
===============================================================================
"""
