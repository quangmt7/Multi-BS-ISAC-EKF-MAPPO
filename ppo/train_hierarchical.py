"""Command-line entry point for Hierarchical ISAC (SAC + DQN) training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

# Preserve direct invocation with ``python ppo/train_hierarchical.py``.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Hierarchical ISAC training."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Hierarchical ISAC (SAC + DQN)")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=project_root / "config",
        help="Directory containing config YAMLs",
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override total episodes"
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Override subframes per episode",
    )
    parser.add_argument(
        "--subframes-per-frame",
        type=int,
        default=None,
        help="Override subframes per frame (F)",
    )
    parser.add_argument(
        "--single-bwp",
        action="store_true",
        help="Run Baseline 2.2 Hierarchical-singleBWP",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results",
        help="Custom output directory",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate figures from an existing metrics file without training again",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and start Hierarchical ISAC training."""
    args = parse_args(argv)
    from ppo.utils.hier_trainer import HierarchicalISACTrainer
    from ppo.utils.project_config import ProjectConfig
    from ppo.utils.result_artifacts import ResultArtifacts

    run_name = "hierarchical_single_bwp" if args.single_bwp else "hierarchical"
    if args.figures_only:
        artifacts = ResultArtifacts(args.output_dir, run_name=run_name)
        paths = artifacts.render_existing()
        print(f"Created {len(paths)} figures in {artifacts.figures_dir}")
        return

    merged_config = ProjectConfig.from_directory(
        args.config_dir, mode="hierarchical"
    ).data

    trainer = HierarchicalISACTrainer(
        config=merged_config,
        subframes_per_frame=args.subframes_per_frame,
        single_bwp_mode=args.single_bwp,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )

    mode_str = "Single-BWP Benchmark" if args.single_bwp else "Multi-BWP Proposed"
    print(f"=== Starting Hierarchical ISAC Training ({mode_str}) ===")
    trainer.run(total_episodes=args.episodes, episode_length=args.episode_length)


if __name__ == "__main__":
    main()

"""
===============================================================================
USAGE GUIDE / CLI CHEATSHEET (train_hierarchical.py):
===============================================================================

1. TRAIN PROPOSED MULTI-BWP HIERARCHICAL ISAC (SAC + DQN):
   python ppo/train_hierarchical.py

2. TRAIN BENCHMARK SINGLE-BWP HIERARCHICAL ISAC:
   python ppo/train_hierarchical.py --single-bwp

3. OVERRIDE EPISODES AND TIMESCALE PARAMETERS:
   python ppo/train_hierarchical.py --episodes 500
   python ppo/train_hierarchical.py --episodes 300 --episode-length 100 --subframes-per-frame 10

4. RESUME TRAINING FROM A CHECKPOINT:
   python ppo/train_hierarchical.py --resume results/checkpoints/hierarchical/episode_000050.pt
   python ppo/train_hierarchical.py --single-bwp --resume results/checkpoints/hierarchical_single_bwp/episode_000050.pt

5. RE-GENERATE FIGURES WITHOUT RETRAINING:
   python ppo/train_hierarchical.py --figures-only
   python ppo/train_hierarchical.py --single-bwp --figures-only

6. CUSTOM CONFIGURATION & OUTPUT DIRECTORIES:
   python ppo/train_hierarchical.py --config-dir custom_config/ --output-dir custom_results/
===============================================================================
"""
