"""Command-line entry point for the multi-BS ISAC MAPPO baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

# Preserve direct invocation with ``python ppo/train.py``.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MAPPO training.

    Argument:
        (None).

    Return:
        (arguments) (argparse.Namespace): Parsed command-line arguments.
    """

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the multi-BS ISAC MAPPO agent")
    parser.add_argument("--config-dir", type=Path, default=project_root / "config")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Generate figures from an existing metrics file without training again",
    )
    return parser.parse_args()


