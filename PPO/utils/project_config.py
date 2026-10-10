"""Load and validate MAPPO project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


class ProjectConfig:
    """Load and validate the configuration used by MAPPO training.

    Argument:
        (data) (Mapping[str, Any]): Merged project configuration.

    Return:
        (None).

    Note:
        Configuration files are kept separate by concern and may not overwrite
        the same top-level section.
    """

    FILE_NAMES = ("system.yaml", "efk.yaml", "mappo.yaml")

    def __init__(self, data: Mapping[str, Any]):
        """Store a configuration mapping.

        Argument:
            (data) (Mapping[str, Any]): Merged project configuration.

        Return:
            (None).
        """

        self.data = dict(data)

    @classmethod
    def from_directory(cls, config_dir: str | Path) -> ProjectConfig:
        """Load the project configuration from a directory.

        Argument:
            (config_dir) (str | Path): Directory containing the YAML files.

        Return:
            (project_config) (ProjectConfig): Loaded and validated configuration.
        """

        config_dir = Path(config_dir)
        merged: dict[str, Any] = {}

        for filename in cls.FILE_NAMES:
            path = config_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing configuration file: {path}")

            with path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream) or {}

            if not isinstance(document, dict):
                raise ValueError(f"{path} must contain a YAML mapping at its root")

            duplicates = merged.keys() & document.keys()
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise ValueError(f"Duplicate top-level config section(s): {names}")
            merged.update(document)

        project_config = cls(merged)
        project_config.validate()
        return project_config

    def validate(self) -> None:
        """Validate fields required by the training and agent contracts.

        Argument:
            (None).

        Return:
            (None).
        """

        for section in ("system_parameters", "ekf_parameters", "mappo_parameters"):
            if not isinstance(self.data.get(section), Mapping):
                raise ValueError(f"Missing or invalid config section: {section}")

        system = self.data["system_parameters"]
        for name in ("num_bs", "num_ue", "num_target", "num_prb"):
            value = system.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"system_parameters.{name} must be a positive integer")

        mappo = self.data["mappo_parameters"]
        training = mappo.get("training")
        if not isinstance(training, Mapping):
            raise ValueError(
                "Missing or invalid config section: mappo_parameters.training"
            )

        for name in ("total_episodes", "episode_length"):
            value = training.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"mappo_parameters.training.{name} must be a positive integer"
                )
        if training["episode_length"] < 2:
            raise ValueError(
                "mappo_parameters.training.episode_length must be at least 2 because "
                "the current agent normalizes rollout advantages"
            )

        for name in ("save_interval", "eval_interval"):
            value = training.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"mappo_parameters.training.{name} must be a non-negative integer"
                )

        log_interval = training.get("log_interval", 1)
        if (
            isinstance(log_interval, bool)
            or not isinstance(log_interval, int)
            or log_interval <= 0
        ):
            raise ValueError(
                "mappo_parameters.training.log_interval must be a positive integer"
            )
        if isinstance(training.get("seed", 0), bool) or not isinstance(
            training.get("seed", 0), int
        ):
            raise ValueError("mappo_parameters.training.seed must be an integer")
        device = training.get("device", "cpu")
        if device not in ("cpu", "cuda"):
            raise ValueError(
                "mappo_parameters.training.device must be either 'cpu' or 'cuda'"
            )

        ppo = mappo.get("ppo")
        if not isinstance(ppo, Mapping):
            raise ValueError("Missing or invalid config section: mappo_parameters.ppo")

        for name in (
            "lr_actor",
            "lr_critic",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "max_grad_norm",
        ):
            value = ppo.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"mappo_parameters.ppo.{name} must be positive")

        for name in ("ppo_epochs", "batch_size"):
            value = ppo.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"mappo_parameters.ppo.{name} must be a positive integer"
                )

        for name in ("entropy_coef", "value_loss_coef"):
            value = ppo.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"mappo_parameters.ppo.{name} must be non-negative")

        if not 0 < float(ppo["gamma"]) <= 1:
            raise ValueError("mappo_parameters.ppo.gamma must be in (0, 1]")
        if not 0 < float(ppo["gae_lambda"]) <= 1:
            raise ValueError("mappo_parameters.ppo.gae_lambda must be in (0, 1]")
        if not 0 < float(ppo["clip_epsilon"]) < 1:
            raise ValueError("mappo_parameters.ppo.clip_epsilon must be in (0, 1)")

        # The agreed baseline uses PPO clipping for Actor and MSE for Critic.
        for name in ("use_clipped_value_loss", "use_huber_loss"):
            value = ppo.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"mappo_parameters.ppo.{name} must be Boolean")
        if ppo.get("use_clipped_value_loss", False):
            raise ValueError(
                "Baseline requires MSE critic loss without value clipping; set "
                "mappo_parameters.ppo.use_clipped_value_loss to false"
            )
        if ppo.get("use_huber_loss", False):
            raise ValueError(
                "Baseline requires MSE critic loss; set "
                "mappo_parameters.ppo.use_huber_loss to false"
            )

        action_space = mappo.get("action_space", {})
        if not isinstance(action_space, Mapping):
            raise ValueError(
                "Missing or invalid config section: mappo_parameters.action_space"
            )
        expected_choices = 1 + system["num_target"] + system["num_ue"]
        configured_choices = action_space.get("num_discrete_choices", expected_choices)
        if configured_choices != expected_choices:
            raise ValueError(
                "mappo_parameters.action_space.num_discrete_choices must equal "
                f"1 + num_target + num_ue = {expected_choices}"
            )
        if action_space.get("power_action_type", "continuous") != "continuous":
            raise ValueError(
                "mappo_parameters.action_space.power_action_type must be 'continuous'"
            )

        rollout_threads = training.get("rollout_threads", 1)
        if isinstance(rollout_threads, bool) or not isinstance(rollout_threads, int):
            raise ValueError(
                "mappo_parameters.training.rollout_threads must be an integer"
            )
        if rollout_threads != 1:
            raise ValueError(
                "This trainer currently supports exactly one rollout thread"
            )
