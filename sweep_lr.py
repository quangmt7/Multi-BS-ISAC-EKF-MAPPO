"""Compare MAPPO actor and critic learning rates on fixed scenarios."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ppo.utils.learning_rate_sweep import LearningRateSweep


def learning_rate_pair(value: str) -> tuple[float, float]:
    """Parse one actor:critic learning-rate pair.

    Argument:
        (value) (str): Pair written as actor_rate:critic_rate.

    Return:
        (pair) (tuple[float, float]): Positive finite learning rates.
    """
    try:
        actor_lr, critic_lr = (float(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use ACTOR_LR:CRITIC_LR") from exc
    if not all(math.isfinite(rate) and rate > 0 for rate in (actor_lr, critic_lr)):
        raise argparse.ArgumentTypeError("Learning rates must be positive and finite")
    return actor_lr, critic_lr


def main() -> None:
    """Parse trial settings and run the learning-rate comparison.

    Argument:
        (None).

    Return:
        (None).
    """
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=root / "config")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--pairs", type=learning_rate_pair, nargs="+",
        default=[(1e-4, 3e-4), (3e-5, 3e-4), (3e-4, 3e-4),
                 (1e-4, 1e-4), (1e-4, 1e-3)],
    )
    args = parser.parse_args()
    LearningRateSweep(
        args.config_dir, args.output_root, args.pairs, args.seeds, args.episodes
    ).run()


if __name__ == "__main__":
    main()
