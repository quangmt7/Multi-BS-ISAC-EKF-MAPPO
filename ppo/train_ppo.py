"""Command-line entry point for multi-BS ISAC MAPPO training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

# Preserve direct invocation with ``python ppo/train_ppo.py``.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for MAPPO training."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the multi-BS ISAC MAPPO agent")
    parser.add_argument("--config-dir", type=Path, default=project_root / "config")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--single-bwp",
        action="store_true",
        help="Restrict each slice/target per BS to at most one BWP",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate figures from an existing metrics file without training again",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and start MAPPO training."""
    args = parse_args(argv)
    from ppo.utils.mappo_trainer import MAPPOTrainer
    from ppo.utils.project_config import ProjectConfig
    from ppo.utils.result_artifacts import ResultArtifacts

    run_name = "mappo_single_bwp" if args.single_bwp else "mappo"
    if args.figures_only:
        artifacts = ResultArtifacts(args.output_dir, run_name=run_name)
        paths = artifacts.render_existing()
        print(f"Created {len(paths)} figures in {artifacts.figures_dir}")
        return

    config = ProjectConfig.from_directory(args.config_dir).data
    trainer = MAPPOTrainer(
        config,
        output_dir=args.output_dir,
        resume_from=args.resume,
        single_bwp_mode=args.single_bwp,
    )
    trainer.run()


if __name__ == "__main__":
    main()

"""
===============================================================================
USAGE GUIDE / CLI CHEATSHEET (train_ppo.py):
===============================================================================

1. TRAIN PROPOSED MULTI-BWP MAPPO (Default):
   python ppo/train_ppo.py

2. TRAIN BENCHMARK SINGLE-BWP MAPPO:
   python ppo/train_ppo.py --single-bwp

3. RESUME TRAINING FROM A CHECKPOINT:
   python ppo/train_ppo.py --resume results/checkpoints/mappo/episode_000100.pt
   python ppo/train_ppo.py --single-bwp --resume results/checkpoints/mappo_single_bwp/episode_000100.pt

4. RE-GENERATE FIGURES WITHOUT RETRAINING:
   python ppo/train_ppo.py --figures-only
   python ppo/train_ppo.py --single-bwp --figures-only

5. CUSTOM CONFIGURATION & OUTPUT DIRECTORIES:
   python ppo/train_ppo.py --config-dir custom_config/ --output-dir custom_results/
===============================================================================
"""
