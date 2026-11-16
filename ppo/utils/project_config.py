"""Load and validate MAPPO and Hierarchical ISAC project configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


class ProjectConfig:
    """Load and validate the configuration used by MAPPO or Hierarchical training.

    Argument:
        (data) (Mapping[str, Any]): Merged project configuration.
        (mode) (str): Validation mode, ``"mappo"`` (default) or ``"hierarchical"``.

    Return:
        (None).

    Note:
        Configuration files are kept separate by concern and may not overwrite
        the same top-level section.
    """

    FILE_NAMES = ("system.yaml", "ekf.yaml", "mappo.yaml")
    HIERARCHICAL_FILE_NAMES = ("system.yaml", "ekf.yaml", "mappo.yaml", "hierarchical.yaml")
    SUPPORTED_MODES = ("mappo", "hierarchical")

    _SAC_REWARD_KEYS = frozenset({"alpha", "beta", "gamma", "lambda_qos"})
    _DQN_REWARD_KEYS = frozenset(
        {"alpha", "beta", "gamma", "lambda_qos", "k1", "k2", "k3"}
    )

    def __init__(self, data: Mapping[str, Any], mode: str = "mappo"):
        """Store a configuration mapping and validation mode.

        Argument:
            (data) (Mapping[str, Any]): Merged project configuration.
            (mode) (str): Validation mode (``"mappo"`` or ``"hierarchical"``).

        Return:
            (None).
        """

        self._validate_mode(mode)
        self.data = dict(data)
        self.mode = mode

    @classmethod
    def _validate_mode(cls, mode: str) -> str:
        """Ensure the requested configuration mode is supported."""

        if mode not in cls.SUPPORTED_MODES:
            supported = ", ".join(repr(item) for item in cls.SUPPORTED_MODES)
            raise ValueError(
                f"Unsupported config mode {mode!r}; expected one of {supported}"
            )
        return mode

    @classmethod
    def from_directory(
        cls, config_dir: str | Path, mode: str = "mappo"
    ) -> ProjectConfig:
        """Load the project configuration from a directory.

        Argument:
            (config_dir) (str | Path): Directory containing the YAML files.
            (mode) (str): Validation mode (``"mappo"`` or ``"hierarchical"``).

        Return:
            (project_config) (ProjectConfig): Loaded and validated configuration.
        """

        cls._validate_mode(mode)
        config_dir = Path(config_dir)
        merged: dict[str, Any] = {}
        filenames = (
            cls.FILE_NAMES if mode == "mappo" else cls.HIERARCHICAL_FILE_NAMES
        )

        for filename in filenames:
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

        project_config = cls(merged, mode=mode)
        project_config.validate()
        return project_config

    @staticmethod
    def _require_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    @classmethod
    def _require_positive_int(cls, value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @classmethod
    def _require_non_negative_int(cls, value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return int(value)

    @staticmethod
    def _require_finite_number(
        value: Any, name: str, *, min_value: float | None = None, strictly_positive: bool = False
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
        numeric = float(value)
        if strictly_positive and numeric <= 0.0:
            raise ValueError(f"{name} must be positive")
        if min_value is not None and numeric < min_value:
            raise ValueError(f"{name} must be non-negative" if min_value == 0.0 else f"{name} must be >= {min_value}")
        return numeric

    def _validate_interference(
        self, system: Mapping[str, Any], mappo: Mapping[str, Any] | None = None
    ) -> None:
        """Validate canonical interference settings and legacy duplicates.

        Argument:
            (system) (Mapping[str, Any]): Physical system settings.
            (mappo) (Mapping[str, Any] | None): Optional MAPPO settings with
                legacy copies.

        Return:
            (None).

        Note:
            The channel reads system_parameters.channel_and_interference.
            Conflicting legacy copies are rejected instead of being ignored.
        """

        channel = system.get("channel_and_interference", {})
        legacy = mappo.get("interference", {}) if mappo is not None else {}
        if not isinstance(channel, Mapping) or not isinstance(legacy, Mapping):
            raise ValueError("Interference settings must be mappings")
        for name in ("c2s_leakage_eta0", "c2s_leakage_alpha", "ibi_coupling_eta"):
            try:
                value = float(channel[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"system_parameters.channel_and_interference.{name} must be numeric"
                ) from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name in legacy:
                try:
                    duplicate = float(legacy[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"mappo_parameters.interference.{name} must be numeric"
                    ) from exc
                if not np.isclose(value, duplicate, rtol=1e-6, atol=0.0):
                    raise ValueError(
                        f"Conflicting {name}: use system_parameters.channel_and_interference "
                        "as the source of truth and remove the legacy MAPPO copy"
                    )

    def _validate_system_common(self, system: Mapping[str, Any]) -> None:
        """Validate core system, BWP, cost, and EKF timing fields shared across modes."""

        for name in ("num_bs", "num_target", "num_embb", "num_urllc",
                     "num_ue_1embb", "num_ue_1urllc"):
            value = system.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"system_parameters.{name} must be a positive integer")

        bwp = system.get("bwp")
        if not isinstance(bwp, list) or not bwp or any(
            isinstance(mu, bool) or not isinstance(mu, int) or mu < 0 for mu in bwp
        ):
            raise ValueError("system_parameters.bwp must list non-negative numerologies")
        for name in ("nPrb", "pEach", "fStart"):
            values = system.get(name)
            if not isinstance(values, list) or len(values) != len(bwp):
                raise ValueError(f"system_parameters.{name} must have one entry per BWP")
            for value in values:
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not np.isfinite(value)):
                    raise ValueError(f"system_parameters.{name} must contain finite numbers")
                if name == "nPrb" and (not isinstance(value, int) or value <= 0):
                    raise ValueError("system_parameters.nPrb must contain positive integers")
                if name == "pEach" and value <= 0:
                    raise ValueError("system_parameters.pEach must contain positive powers (mW)")
        for name in ("prbBW", "slot_duration_ms", "urllc_packet_size_bits"):
            value = system.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"system_parameters.{name} must be explicitly set and positive")
        for name in ("cost_switch_c", "cost_switch_s", "cost_gb"):
            value = system.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"system_parameters.{name} must be finite and non-negative")
        for name in ("pa_eff", "dt"):
            value = system.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"system_parameters.{name} must be finite and positive")
        if system["pa_eff"] > 1:
            raise ValueError("system_parameters.pa_eff must not exceed 1")
        if not np.isclose(system["dt"], system["slot_duration_ms"] / 1000., rtol=1e-10, atol=0):
            raise ValueError("system_parameters.dt must match slot_duration_ms in seconds")
        ekf_dt = self.data.get("ekf_parameters", {}).get("dt")
        if (isinstance(ekf_dt, bool) or not isinstance(ekf_dt, (int, float))
                or not np.isfinite(ekf_dt)
                or not np.isclose(ekf_dt, system["dt"], rtol=1e-10, atol=0)):
            raise ValueError("ekf_parameters.dt must match slot_duration_ms in seconds")
        sim = system.get("simulation")
        if not isinstance(sim, Mapping):
            raise ValueError("Missing or invalid config section: system_parameters.simulation")
        self._require_positive_int(
            sim.get("num_slots_per_episode"),
            "system_parameters.simulation.num_slots_per_episode",
        )
        self._require_positive_int(
            sim.get("num_subframes_per_frame"),
            "system_parameters.simulation.num_subframes_per_frame",
        )

        req = system.get("requirements")
        if not isinstance(req, Mapping):
            raise ValueError("Missing or invalid config section: system_parameters.requirements")
        for name in ("min_ue_rate_mbps", "max_ue_delay"):
            self._require_finite_number(
                req.get(name), f"system_parameters.requirements.{name}", strictly_positive=True
            )
        for name in ("min_sensing_success_prob", "min_detection_prob"):
            prob = self._require_finite_number(
                req.get(name), f"system_parameters.requirements.{name}", strictly_positive=True
            )
            if prob > 1.0:
                raise ValueError(f"system_parameters.requirements.{name} must be in (0, 1]")
        p_fa = self._require_finite_number(
            req.get("max_prob_false_alarm"),
            "system_parameters.requirements.max_prob_false_alarm",
            strictly_positive=True,
        )
        if p_fa >= 1.0:
            raise ValueError("system_parameters.requirements.max_prob_false_alarm must be in (0, 1)")

        scenario = system.get("scenario")
        if not isinstance(scenario, Mapping):
            raise ValueError("Missing or invalid config section: system_parameters.scenario")
        for name in ("area_size_x_m", "area_size_y_m", "area_size_z_m"):
            self._require_finite_number(
                scenario.get(name), f"system_parameters.scenario.{name}", strictly_positive=True
            )

        sinr = system.get("sinr")
        if not isinstance(sinr, Mapping):
            raise ValueError("Missing or invalid config section: system_parameters.sinr")
        for name in ("gamma_c_min_linear", "gamma_s_min_linear"):
            gamma_val = self._require_finite_number(
                sinr.get(name), f"system_parameters.sinr.{name}"
            )
            if 1.0 + gamma_val <= 0.0:
                raise ValueError(f"system_parameters.sinr.{name} must satisfy 1 + gamma > 0")

        ekf = self.data.get("ekf_parameters")
        if not isinstance(ekf, Mapping):
            raise ValueError("Missing or invalid config section: ekf_parameters")
        state_dim = self._require_positive_int(ekf.get("state_dim"), "ekf_parameters.state_dim")
        self._require_positive_int(ekf.get("measurement_dim"), "ekf_parameters.measurement_dim")
        limits = ekf.get("uncertainty_limits")
        if not isinstance(limits, Mapping):
            raise ValueError("Missing or invalid config section: ekf_parameters.uncertainty_limits")
        for key in ("u_pos_max", "u_vel_max"):
            self._require_finite_number(
                limits.get(key),
                f"ekf_parameters.uncertainty_limits.{key}",
                strictly_positive=True,
            )
        init_cov = ekf.get("initial_covariance")
        if not isinstance(init_cov, Mapping) or not init_cov:
            raise ValueError("Missing or invalid config section: ekf_parameters.initial_covariance")
        if "p_diag_init" not in init_cov:
            raise ValueError("ekf_parameters.initial_covariance.p_diag_init is required")
        for key, val in init_cov.items():
            if isinstance(val, (list, tuple)):
                if not val:
                    raise ValueError(f"ekf_parameters.initial_covariance.{key} must not be empty")
                if key == "p_diag_init" and len(val) != state_dim:
                    raise ValueError(
                        f"ekf_parameters.initial_covariance.p_diag_init must have length {state_dim}"
                    )
                for idx, item in enumerate(val):
                    self._require_finite_number(
                        item,
                        f"ekf_parameters.initial_covariance.{key}[{idx}]",
                        strictly_positive=True,
                    )
            else:
                self._require_finite_number(
                    val,
                    f"ekf_parameters.initial_covariance.{key}",
                    strictly_positive=True,
                )

    def _validate_reward_and_interference(
        self, system: Mapping[str, Any], mappo: Mapping[str, Any]
    ) -> None:
        """Validate physical reward weights, shaping, normalization, penalties, and interference."""

        for section, names in (
            ("reward_weights", ("alpha_aoi", "beta_uncertainty", "gamma_power", "lambda_service")),
            ("reward_shaping", ("k_aoi", "k_uncertainty", "k_service")),
        ):
            sub = mappo.get(section)
            if not isinstance(sub, Mapping):
                raise ValueError(f"Missing or invalid config section: mappo_parameters.{section}")
            for name in names:
                value = sub.get(name)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not np.isfinite(value) or value < 0
                        or (section == "reward_shaping" and value == 0)):
                    raise ValueError(f"mappo_parameters.{section}.{name} must be finite and "
                                     + ("positive" if section == "reward_shaping" else "non-negative"))

        normalization = mappo.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("Missing or invalid config section: mappo_parameters.normalization")
        self._require_finite_number(
            normalization.get("a_max"),
            "mappo_parameters.normalization.a_max",
            strictly_positive=True,
        )

        penalties = mappo.get("penalties")
        if not isinstance(penalties, Mapping):
            raise ValueError("Missing or invalid config section: mappo_parameters.penalties")
        for key, value in penalties.items():
            self._require_finite_number(
                value, f"mappo_parameters.penalties.{key}", min_value=0.0
            )

        self._validate_interference(system, mappo)

    def _validate_training_schedule(
        self, training: Mapping[str, Any], prefix: str
    ) -> None:
        """Validate episode schedule, logging, seed, and device fields."""

        for name in ("total_episodes", "episode_length"):
            self._require_positive_int(training.get(name), f"{prefix}.{name}")

        self._require_non_negative_int(
            training.get("save_interval", 0), f"{prefix}.save_interval"
        )
        self._require_positive_int(
            training.get("log_interval", 1), f"{prefix}.log_interval"
        )
        self._require_int(training.get("seed", 0), f"{prefix}.seed")

        device = training.get("device", "cpu")
        if device not in ("cpu", "cuda"):
            raise ValueError(f"{prefix}.device must be either 'cpu' or 'cuda'")

    def validate(self, mode: str | None = None) -> None:
        """Validate fields required by the training and agent contracts.

        Argument:
            (mode) (str | None): Optional mode override (defaults to ``self.mode``).

        Return:
            (None).
        """

        active_mode = self._validate_mode(self.mode if mode is None else mode)
        if active_mode == "mappo":
            self._validate_mappo()
        else:
            self._validate_hierarchical()

    def _validate_mappo(self) -> None:
        """Validate configuration sections required by MAPPO training."""

        for section in ("system_parameters", "ekf_parameters", "mappo_parameters"):
            if not isinstance(self.data.get(section), Mapping):
                raise ValueError(f"Missing or invalid config section: {section}")

        system = self.data["system_parameters"]
        self._validate_system_common(system)
        mappo = self.data["mappo_parameters"]
        self._validate_reward_and_interference(system, mappo)
        training = mappo.get("training")
        if not isinstance(training, Mapping):
            raise ValueError(
                "Missing or invalid config section: mappo_parameters.training"
            )

        self._validate_training_schedule(training, "mappo_parameters.training")
        self._require_positive_int(
            training.get("update_interval", 1),
            "mappo_parameters.training.update_interval",
        )
        self._require_non_negative_int(
            training.get("eval_interval", 0),
            "mappo_parameters.training.eval_interval",
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
            self._require_positive_int(ppo.get(name), f"mappo_parameters.ppo.{name}")

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

        # Use PPO clipping for the actor and MSE loss for the critic.
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
        num_ue = (system["num_embb"] * system["num_ue_1embb"]
                  + system["num_urllc"] * system["num_ue_1urllc"])
        expected_choices = 1 + system["num_target"] + num_ue
        configured_choices = action_space.get("num_discrete_choices", expected_choices)
        if configured_choices != expected_choices:
            raise ValueError(
                "mappo_parameters.action_space.num_discrete_choices must equal "
                f"1 + num_target + total slice UEs = {expected_choices}"
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

        network = mappo.get("network")
        if not isinstance(network, Mapping):
            raise ValueError("Missing or invalid config section: mappo_parameters.network")
        for role in ("actor", "critic"):
            role_cfg = network.get(role)
            if not isinstance(role_cfg, Mapping):
                raise ValueError(
                    f"Missing or invalid config section: mappo_parameters.network.{role}"
                )
            hidden_dims = role_cfg.get("hidden_dims")
            if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) != 2:
                raise ValueError(
                    f"mappo_parameters.network.{role}.hidden_dims must contain exactly 2 positive integers"
                )
            for idx, dim in enumerate(hidden_dims):
                self._require_positive_int(
                    dim, f"mappo_parameters.network.{role}.hidden_dims[{idx}]"
                )

    def _validate_hierarchical(self) -> None:
        """Validate configuration sections required by Hierarchical ISAC training."""

        for section in (
            "system_parameters",
            "ekf_parameters",
            "mappo_parameters",
            "hierarchical_parameters",
        ):
            if not isinstance(self.data.get(section), Mapping):
                raise ValueError(f"Missing or invalid config section: {section}")

        system = self.data["system_parameters"]
        self._validate_system_common(system)
        self._validate_reward_and_interference(system, self.data["mappo_parameters"])

        hier = self.data["hierarchical_parameters"]
        timescale = hier.get("timescale")
        if not isinstance(timescale, Mapping):
            raise ValueError(
                "Missing or invalid config section: hierarchical_parameters.timescale"
            )
        subframes_per_frame = self._require_positive_int(
            timescale.get("subframes_per_frame"),
            "hierarchical_parameters.timescale.subframes_per_frame",
        )
        sim_subframes = system["simulation"]["num_subframes_per_frame"]
        if subframes_per_frame != sim_subframes:
            raise ValueError(
                "hierarchical_parameters.timescale.subframes_per_frame "
                f"({subframes_per_frame}) must match "
                f"system_parameters.simulation.num_subframes_per_frame ({sim_subframes})"
            )

        if "training" in hier:
            training = hier["training"]
            if not isinstance(training, Mapping):
                raise ValueError(
                    "Invalid config section: hierarchical_parameters.training"
                )
            self._validate_training_schedule(
                training, "hierarchical_parameters.training"
            )
        else:
            self._validate_training_schedule(
                timescale, "hierarchical_parameters.timescale"
            )

        self._validate_hierarchical_sac(hier)
        self._validate_hierarchical_dqn(hier)
        if "reward_weights" in hier:
            self._validate_hierarchical_reward_weights(hier["reward_weights"])

    @classmethod
    def _validate_hierarchical_sac(cls, hier: Mapping[str, Any]) -> None:
        sac = hier.get("sac")
        if not isinstance(sac, Mapping):
            raise ValueError(
                "Missing or invalid config section: hierarchical_parameters.sac"
            )
        prefix = "hierarchical_parameters.sac"

        for name in ("actor_lr", "critic_lr", "alpha_lr"):
            cls._require_finite_number(
                sac.get(name), f"{prefix}.{name}", strictly_positive=True
            )

        for name in ("gamma", "tau"):
            val = cls._require_finite_number(
                sac.get(name), f"{prefix}.{name}", strictly_positive=True
            )
            if val > 1.0:
                raise ValueError(f"{prefix}.{name} must be in (0, 1]")

        for name in ("batch_size", "replay_capacity"):
            cls._require_positive_int(sac.get(name), f"{prefix}.{name}")

        for opt_int in ("policy_delay", "update_after", "updates_per_frame", "hidden_dim"):
            if opt_int in sac:
                cls._require_positive_int(sac[opt_int], f"{prefix}.{opt_int}")

        for opt_pos_float in ("init_alpha", "action_scale"):
            if opt_pos_float in sac:
                cls._require_finite_number(
                    sac[opt_pos_float], f"{prefix}.{opt_pos_float}", strictly_positive=True
                )
        if "action_bias" in sac:
            cls._require_finite_number(sac["action_bias"], f"{prefix}.action_bias")
        if "auto_entropy" in sac and not isinstance(sac["auto_entropy"], bool):
            raise ValueError(f"{prefix}.auto_entropy must be Boolean")
        if "target_entropy" in sac:
            te = sac["target_entropy"]
            if te is not None and te != "auto":
                cls._require_finite_number(te, f"{prefix}.target_entropy")

    @classmethod
    def _validate_hierarchical_dqn(cls, hier: Mapping[str, Any]) -> None:
        dqn = hier.get("dqn")
        if not isinstance(dqn, Mapping):
            raise ValueError(
                "Missing or invalid config section: hierarchical_parameters.dqn"
            )
        prefix = "hierarchical_parameters.dqn"

        cls._require_finite_number(dqn.get("lr"), f"{prefix}.lr", strictly_positive=True)
        gamma = cls._require_finite_number(
            dqn.get("gamma"), f"{prefix}.gamma", strictly_positive=True
        )
        if gamma > 1.0:
            raise ValueError(f"{prefix}.gamma must be in (0, 1]")

        for name in ("batch_size", "target_update_freq", "replay_capacity"):
            cls._require_positive_int(dqn.get(name), f"{prefix}.{name}")

        if "hidden_dim" in dqn:
            cls._require_positive_int(dqn["hidden_dim"], f"{prefix}.hidden_dim")

        eps_start = cls._require_finite_number(
            dqn.get("epsilon_start"), f"{prefix}.epsilon_start", min_value=0.0
        )
        eps_min = cls._require_finite_number(
            dqn.get("epsilon_min"), f"{prefix}.epsilon_min", min_value=0.0
        )
        if eps_start > 1.0:
            raise ValueError(f"{prefix}.epsilon_start must be in [0, 1]")
        if eps_min > 1.0:
            raise ValueError(f"{prefix}.epsilon_min must be in [0, 1]")
        if eps_min > eps_start:
            raise ValueError(
                f"{prefix}.epsilon_min must be <= {prefix}.epsilon_start"
            )
        cls._require_finite_number(
            dqn.get("epsilon_decay"), f"{prefix}.epsilon_decay", strictly_positive=True
        )

    @classmethod
    def _validate_hierarchical_reward_weights(cls, reward_weights: Any) -> None:
        prefix = "hierarchical_parameters.reward_weights"
        if not isinstance(reward_weights, Mapping):
            raise ValueError(f"Invalid config section: {prefix}")
        expected_sections = {"sac", "dqn"}
        if set(reward_weights.keys()) != expected_sections:
            raise ValueError(
                f"{prefix} must contain exact sections {sorted(expected_sections)}"
            )

        for section_name, expected_keys in (
            ("sac", cls._SAC_REWARD_KEYS),
            ("dqn", cls._DQN_REWARD_KEYS),
        ):
            section = reward_weights[section_name]
            if not isinstance(section, Mapping):
                raise ValueError(f"Invalid config section: {prefix}.{section_name}")
            if set(section.keys()) != expected_keys:
                raise ValueError(
                    f"{prefix}.{section_name} must contain exact keys "
                    f"{sorted(expected_keys)}"
                )
            for key in expected_keys:
                cls._require_finite_number(
                    section[key],
                    f"{prefix}.{section_name}.{key}",
                    min_value=0.0,
                )
