"""Run comparable actor and critic learning-rate trials."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import mean

from ppo.utils.mappo_trainer import MAPPOTrainer
from ppo.utils.project_config import ProjectConfig


class LearningRateSweep:
    """Train each learning-rate pair with the same scenario seeds.

    Argument:
        (config_dir) (Path): Directory containing the project YAML files.
        (output_root) (Path): Directory for separate trial results.
        (pairs) (list[tuple[float, float]]): Actor and critic learning rates.
        (seeds) (list[int]): Training seeds shared by all pairs.
        (episodes) (int): Number of episodes per trial.

    Return:
        (None).
    """

    METRICS = (
        "episode_return",
        "comm_rate_mean_mbps",
        "qos_satisfaction_rate",
        "sensing_success_prob_mean",
        "u_pos_mean",
        "u_vel_mean",
        "aoi_mean",
        "norm_power",
        "episode_length",
    )

    def __init__(
        self,
        config_dir: Path,
        output_root: Path,
        pairs: list[tuple[float, float]],
        seeds: list[int],
        episodes: int,
    ) -> None:
        """Load the fixed baseline and validate the trial dimensions.

        Argument:
            (config_dir) (Path): Project configuration directory.
            (output_root) (Path): Parent directory for all trials.
            (pairs) (list[tuple[float, float]]): Learning-rate pairs.
            (seeds) (list[int]): Repeated random seeds.
            (episodes) (int): Episodes per trial.

        Return:
            (None).
        """
        if not pairs or not seeds or episodes <= 0:
            raise ValueError("Provide at least one pair, one seed, and a positive episode count")
        self.baseline = ProjectConfig.from_directory(config_dir).data
        system_steps = self.baseline["system_parameters"]["simulation"]["num_slots_per_episode"]
        training_steps = self.baseline["mappo_parameters"]["training"]["episode_length"]
        if system_steps != training_steps:
            raise ValueError("Environment and trainer episode lengths must agree")
        self.output_root = output_root
        self.pairs = pairs
        self.seeds = seeds
        self.episodes = episodes

    def run(self) -> list[dict[str, float | int | str]]:
        """Run every trial and save a summary after each completed trial.

        Argument:
            (None).

        Return:
            (summaries) (list[dict[str, float | int | str]]): Trial metrics.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, float | int | str]] = []
        for actor_lr, critic_lr in self.pairs:
            for seed in self.seeds:
                label = self._label(actor_lr, critic_lr, seed)
                trial_dir = self.output_root / label
                if trial_dir.exists():
                    raise FileExistsError(f"Trial output already exists: {trial_dir}")
                config = copy.deepcopy(self.baseline)
                ppo = config["mappo_parameters"]["ppo"]
                training = config["mappo_parameters"]["training"]
                ppo["lr_actor"] = actor_lr
                ppo["lr_critic"] = critic_lr
                training["seed"] = seed
                training["total_episodes"] = self.episodes
                ProjectConfig(config).validate()

                trial_dir.mkdir()
                (trial_dir / "config_snapshot.json").write_text(
                    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Starting {label}", flush=True)
                history = MAPPOTrainer(config, output_dir=trial_dir).run()
                tail = history[-max(1, self.episodes // 5):]
                summary: dict[str, float | int | str] = {
                    "trial": label,
                    "lr_actor": actor_lr,
                    "lr_critic": critic_lr,
                    "seed": seed,
                    "episodes": self.episodes,
                }
                for name in self.METRICS:
                    values = [float(row[name]) for row in tail if name in row]
                    if values:
                        summary[name] = mean(values)
                summaries.append(summary)
                (self.output_root / "summary.json").write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"Finished {label}: {summary}", flush=True)
        return summaries

    @staticmethod
    def _label(actor_lr: float, critic_lr: float, seed: int) -> str:
        """Build a directory name containing the trial parameters.

        Argument:
            (actor_lr) (float): Actor learning rate.
            (critic_lr) (float): Critic learning rate.
            (seed) (int): Training seed.

        Return:
            (label) (str): Directory name for one trial.
        """
        actor_label = format(actor_lr, ".8g").replace(".", "p").replace("-", "m")
        critic_label = format(critic_lr, ".8g").replace(".", "p").replace("-", "m")
        return f"actor_{actor_label}_critic_{critic_label}_seed_{seed}"
