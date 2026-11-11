from ppo.train import parse_args
from ppo.utils.mappo_trainer import MAPPOTrainer
from ppo.utils.project_config import ProjectConfig


def main() -> None:
    """Load configuration and start MAPPO training.

    Argument:
        (None).

    Return:
        (None).
    """
    args = parse_args()
    if args.figures_only:
        from ppo.utils.result_artifacts import ResultArtifacts

        paths = ResultArtifacts(args.output_dir).render_existing()
        print(f"Created {len(paths)} figures in {args.output_dir / 'figures'}")
        return
    config = ProjectConfig.from_directory(args.config_dir).data
    trainer = MAPPOTrainer(
        config,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )
    trainer.run(bwp_mode=True)


if __name__ == "__main__":
    main()
