"""Training entry point for the multi-BS ISAC MAPPO baseline.

The module coordinates training only. State construction, action mapping,
physical models and reward calculation remain responsibilities of the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
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

    FILE_NAMES = ("system.yaml", "ekf.yaml", "mappo.yaml")

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


class RuntimeContract:
    """Validate data exchanged by the environment, agent and rollout buffer.

    Argument:
        (config) (Mapping[str, Any]): Validated project configuration.

    Return:
        (None).
    """

    def __init__(self, config: Mapping[str, Any]):
        """Store dimensions and valid action ranges from the configuration.

        Argument:
            (config) (Mapping[str, Any]): Validated project configuration.

        Return:
            (None).
        """

        system = config["system_parameters"]
        self.num_bs = int(system["num_bs"])
        self.num_prb = int(system["num_prb"])
        self.max_destination = int(system["num_target"] + system["num_ue"])

    def observation_dimensions(
        self, local_observations: Any, global_state: Any
    ) -> tuple[int, int]:
        """Validate observations and return their dimensions.

        Argument:
            (local_observations) (Any)[num_bs, local_obs_dim]: Local BS states.
            (global_state) (Any)[global_state_dim]: Centralized critic state.

        Return:
            (dimensions) (tuple[int, int]): Local and global state dimensions.
        """

        local_array = np.asarray(local_observations, dtype=np.float32)
        global_array = np.asarray(global_state, dtype=np.float32)

        if local_array.ndim != 2:
            raise ValueError(
                "local observations must have shape [num_bs, local_obs_dim]"
            )
        if local_array.shape[0] != self.num_bs:
            raise ValueError(
                f"local observations contain {local_array.shape[0]} agents; "
                f"config num_bs is {self.num_bs}"
            )
        if local_array.shape[1] <= 0:
            raise ValueError("local_obs_dim must be positive")
        if global_array.ndim != 1 or global_array.size == 0:
            raise ValueError("global state must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(local_array)) or not np.all(
            np.isfinite(global_array)
        ):
            raise ValueError("observations must contain only finite values")

        return int(local_array.shape[1]), int(global_array.size)

    def step_result(
        self,
        result: Any,
        expected_local_dim: int,
        expected_global_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        """Validate one transition returned by the environment.

        Argument:
            (result) (Any): Raw value returned by ``env.step``.
            (expected_local_dim) (int): Local state dimension fixed at reset.
            (expected_global_dim) (int): Global state dimension fixed at reset.

        Return:
            (transition) (tuple): Validated next states, reward, flags and info.
        """

        if not isinstance(result, tuple) or len(result) != 6:
            raise ValueError(
                "env.step() must return (local_obs, global_state, reward, "
                "terminated, truncated, info)"
            )

        local_obs, global_state, reward, terminated, truncated, info = result
        actual_local_dim, actual_global_dim = self.observation_dimensions(
            local_obs, global_state
        )
        if (
            actual_local_dim != expected_local_dim
            or actual_global_dim != expected_global_dim
        ):
            raise ValueError(
                "observation dimensions changed during the episode: "
                f"expected ({expected_local_dim}, {expected_global_dim}), "
                f"received ({actual_local_dim}, {actual_global_dim})"
            )

        reward_array = np.asarray(reward)
        if reward_array.ndim != 0 or not np.isfinite(reward_array.item()):
            raise ValueError("env.step() must return one finite scalar team reward")
        if not isinstance(info, Mapping):
            raise ValueError("env.step() info must be a mapping")

        return (
            np.asarray(local_obs, dtype=np.float32),
            np.asarray(global_state, dtype=np.float32),
            float(reward_array.item()),
            self._boolean_flag(terminated, "terminated"),
            self._boolean_flag(truncated, "truncated"),
            dict(info),
        )

    def policy_output(
        self, output: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Validate actions, log-probabilities and value from the agent.

        Argument:
            (output) (Any): Value returned by ``agent.select_action``.

        Return:
            (policy_output) (tuple): Validated PRB actions, power, log-probability
                and centralized value.
        """

        if not isinstance(output, tuple) or len(output) != 4:
            raise ValueError(
                "agent.select_action() must return "
                "(actions_prb, actions_pwr, log_probs, value)"
            )

        actions_prb = np.asarray(output[0])
        actions_pwr = np.asarray(output[1], dtype=np.float32)
        log_probs = np.asarray(output[2], dtype=np.float32)
        value_array = np.asarray(output[3])
        expected_shape = (self.num_bs, self.num_prb)

        if actions_prb.shape != expected_shape:
            raise ValueError(
                f"PRB action shape must be {expected_shape}, got {actions_prb.shape}"
            )
        if not np.issubdtype(actions_prb.dtype, np.integer):
            raise ValueError("PRB actions must use an integer dtype")
        if np.any(actions_prb < 0) or np.any(actions_prb > self.max_destination):
            raise ValueError(f"PRB actions must be in [0, {self.max_destination}]")
        if actions_pwr.shape != expected_shape:
            raise ValueError(
                f"power action shape must be {expected_shape}, got {actions_pwr.shape}"
            )
        if not np.all(np.isfinite(actions_pwr)):
            raise ValueError("power actions contain NaN or Inf")
        if np.any(actions_pwr < 0.0) or np.any(actions_pwr > 1.0):
            raise ValueError(
                "power actions must be normalized to [0, 1] before action mapping"
            )
        if log_probs.shape != (self.num_bs,) or not np.all(np.isfinite(log_probs)):
            raise ValueError(
                f"log-probabilities must be finite with shape ({self.num_bs},)"
            )
        if value_array.ndim != 0 or not np.isfinite(value_array.item()):
            raise ValueError("centralized critic value must be one finite scalar")

        return actions_prb, actions_pwr, log_probs, float(value_array.item())

    def agent_structure(
        self, agent: Any, local_obs_dim: int, global_state_dim: int
    ) -> None:
        """Validate the agent API and neural-network input dimensions.

        Argument:
            (agent) (Any): MAPPO agent instance.
            (local_obs_dim) (int): Local observation dimension from the environment.
            (global_state_dim) (int): Global state dimension from the environment.

        Return:
            (None).
        """

        for method_name in ("select_action", "update"):
            if not callable(getattr(agent, method_name, None)):
                raise TypeError(f"Agent must provide a callable {method_name}()")

        for attribute in (
            "actornetwork",
            "criticnetwork",
            "actor_optimizer",
            "critic_optimizer",
        ):
            if not hasattr(agent, attribute):
                raise TypeError(f"Agent is missing required attribute: {attribute}")

        actor_input = self._first_linear_input(agent.actornetwork)
        critic_input = self._first_linear_input(agent.criticnetwork)
        if actor_input is not None and actor_input != local_obs_dim:
            raise ValueError(
                f"Actor expects {actor_input} inputs, but env returns {local_obs_dim}"
            )
        if critic_input is not None and critic_input != global_state_dim:
            raise ValueError(
                f"Critic expects {critic_input} inputs, but env returns {global_state_dim}"
            )

    def finite_agent_parameters(self, agent: Any) -> None:
        """Reject an update that creates non-finite neural-network parameters.

        Argument:
            (agent) (Any): MAPPO agent instance after an optimization update.

        Return:
            (None).
        """

        for network_name in ("actornetwork", "criticnetwork"):
            network = getattr(agent, network_name, None)
            if not isinstance(network, torch.nn.Module):
                continue
            for parameter_name, parameter in network.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"{network_name}.{parameter_name} contains NaN or Inf"
                    )

    @staticmethod
    def _boolean_flag(value: Any, name: str) -> bool:
        """Convert one scalar Boolean flag without accepting ambiguous values.

        Argument:
            (value) (Any): Candidate Boolean value.
            (name) (str): Field name used in an error message.

        Return:
            (flag) (bool): Validated Boolean value.
        """

        value_array = np.asarray(value)
        if value_array.ndim != 0 or not isinstance(
            value_array.item(), (bool, np.bool_)
        ):
            raise ValueError(f"env.step() {name} must be one Boolean scalar")
        return bool(value_array.item())

    @staticmethod
    def _first_linear_input(network: Any) -> int | None:
        """Read the first linear-layer input dimension of a network.

        Argument:
            (network) (Any): Candidate PyTorch network.

        Return:
            (input_dim) (int | None): Input size, or None for a non-PyTorch object.
        """

        if not isinstance(network, torch.nn.Module):
            return None
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        return None


class CheckpointManager:
    """Persist and restore one MAPPO training run.

    Argument:
        (config) (Mapping[str, Any]): Configuration associated with checkpoints.
        (agent) (Any): MAPPO agent whose state is persisted.

    Return:
        (None).
    """

    def __init__(self, config: Mapping[str, Any], agent: Any):
        """Store the checkpoint dependencies.

        Argument:
            (config) (Mapping[str, Any]): Validated project configuration.
            (agent) (Any): MAPPO agent instance.

        Return:
            (None).
        """

        self.config = dict(config)
        self.agent = agent

    def save(self, path: str | Path, episode: int, global_step: int) -> None:
        """Atomically save model, optimizer and random-generator states.

        Argument:
            (path) (str | Path): Destination checkpoint path.
            (episode) (int): Zero-based episode index stored in the checkpoint.
            (global_step) (int): Number of environment steps already completed.

        Return:
            (None).
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "episode": int(episode),
            "global_step": int(global_step),
            "config": self.config,
            "actor": self.agent.actornetwork.state_dict(),
            "critic": self.agent.criticnetwork.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer": self.agent.critic_optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_random_state"] = torch.cuda.get_rng_state_all()

        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load(self, path: str | Path) -> tuple[int, int]:
        """Restore a compatible checkpoint.

        Argument:
            (path) (str | Path): Existing checkpoint path.

        Return:
            (progress) (tuple[int, int]): Next episode index and global step.

        Note:
            PyTorch checkpoints use pickle internally and must come from a trusted
            project run.
        """

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        device = getattr(self.agent, "device", torch.device("cpu"))
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint root must be a mapping")

        required = (
            "episode",
            "global_step",
            "config",
            "actor",
            "critic",
            "actor_optimizer",
            "critic_optimizer",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"Checkpoint is missing: {', '.join(missing)}")
        if checkpoint["config"] != self.config:
            raise ValueError(
                "Checkpoint configuration differs from the current configuration"
            )

        raw_episode = checkpoint["episode"]
        raw_global_step = checkpoint["global_step"]
        if (
            isinstance(raw_episode, bool)
            or not isinstance(raw_episode, int)
            or raw_episode < 0
        ):
            raise ValueError("Checkpoint episode must be a non-negative integer")
        if (
            isinstance(raw_global_step, bool)
            or not isinstance(raw_global_step, int)
            or raw_global_step < 0
        ):
            raise ValueError("Checkpoint global_step must be a non-negative integer")

        self.agent.actornetwork.load_state_dict(checkpoint["actor"])
        self.agent.criticnetwork.load_state_dict(checkpoint["critic"])
        self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

        return raw_episode + 1, raw_global_step

    @staticmethod
    def prepare_metrics(path: str | Path, next_episode: int) -> str:
        """Align an existing metrics file with resumed checkpoint progress.

        Argument:
            (path) (str | Path): JSON Lines metrics path.
            (next_episode) (int): Zero-based episode index to run next.

        Return:
            (file_mode) (str): File mode used for the resumed training run.
        """

        path = Path(path)
        if next_episode == 0:
            return "w"
        if not path.exists():
            return "a"

        retained: list[str] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid metrics JSON at line {line_number}: {path}"
                    ) from exc
                episode = record.get("episode")
                if isinstance(episode, bool) or not isinstance(episode, int):
                    raise ValueError(
                        f"Invalid episode field at metrics line {line_number}: {path}"
                    )
                if episode <= next_episode:
                    retained.append(json.dumps(record, ensure_ascii=False))

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as stream:
                for line in retained:
                    stream.write(line + "\n")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return "a"


class MAPPOTrainer:
    """Coordinate episodic on-policy MAPPO training.

    Argument:
        (config) (Mapping[str, Any]): Merged project configuration.
        (env) (Any | None): Optional injected environment for verification.
        (agent) (Any | None): Optional injected MAPPO agent.
        (buffer_factory) (Callable[[], Any] | None): Rollout-buffer constructor.
        (output_dir) (str | Path): Directory for metrics and checkpoints.
        (resume_from) (str | Path | None): Optional trusted checkpoint path.

    Return:
        (None).
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        env: Any | None = None,
        agent: Any | None = None,
        buffer_factory: Callable[[], Any] | None = None,
        output_dir: str | Path = "results",
        resume_from: str | Path | None = None,
    ):
        """Initialize dependencies without starting training.

        Argument:
            (config) (Mapping[str, Any]): Merged project configuration.
            (env) (Any | None): Optional injected environment.
            (agent) (Any | None): Optional injected agent.
            (buffer_factory) (Callable[[], Any] | None): Rollout-buffer constructor.
            (output_dir) (str | Path): Result directory.
            (resume_from) (str | Path | None): Optional checkpoint path.

        Return:
            (None).
        """

        project_config = ProjectConfig(config)
        project_config.validate()
        self.config = project_config.data
        self.contract = RuntimeContract(self.config)
        self.env = env
        self.agent = agent
        self.buffer_factory = buffer_factory
        self.output_dir = Path(output_dir)
        self.resume_from = Path(resume_from) if resume_from is not None else None
        self._owns_environment = env is None

    def run(self) -> list[dict[str, Any]]:
        """Run training and return metrics produced in this process.

        Argument:
            (None).

        Return:
            (history) (list[dict[str, Any]]): Episode metrics from this run.
        """

        training = self.config["mappo_parameters"]["training"]
        total_episodes = int(training["total_episodes"])
        episode_length = int(training["episode_length"])
        save_interval = int(training.get("save_interval", 0))
        log_interval = int(training.get("log_interval", 1))
        seed = int(training.get("seed", 0))

        self._seed_everything(seed)
        self.env = self.env if self.env is not None else self._create_environment()

        try:
            initial_local_obs, initial_global_state = self._reset_environment(seed)
            local_obs_dim, global_state_dim = self.contract.observation_dimensions(
                initial_local_obs, initial_global_state
            )
            self._create_missing_agent_components(local_obs_dim, global_state_dim)
            self.contract.agent_structure(self.agent, local_obs_dim, global_state_dim)
            self.contract.finite_agent_parameters(self.agent)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = self.output_dir / "metrics.jsonl"
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_manager = CheckpointManager(self.config, self.agent)

            start_episode = 0
            global_step = 0
            if self.resume_from is not None:
                start_episode, global_step = checkpoint_manager.load(self.resume_from)
            if start_episode > total_episodes:
                raise ValueError(
                    "Checkpoint episode exceeds mappo_parameters.training.total_episodes"
                )

            file_mode = checkpoint_manager.prepare_metrics(metrics_path, start_episode)
            history: list[dict[str, Any]] = []

            with metrics_path.open(file_mode, encoding="utf-8") as metrics_file:
                for episode in range(start_episode, total_episodes):
                    local_obs, global_state = self._reset_environment(seed + episode)
                    reset_dims = self.contract.observation_dimensions(
                        local_obs, global_state
                    )
                    if reset_dims != (local_obs_dim, global_state_dim):
                        raise ValueError(
                            "observation dimensions changed between episodes: "
                            f"expected {(local_obs_dim, global_state_dim)}, "
                            f"received {reset_dims}"
                        )

                    buffer = self.buffer_factory()
                    episode_return = 0.0
                    final_info: dict[str, Any] = {}
                    steps_taken = 0

                    for step in range(episode_length):
                        policy_output = self.contract.policy_output(
                            self.agent.select_action(local_obs, global_state)
                        )
                        actions_prb, actions_pwr, log_probs, value = policy_output
                        transition = self.contract.step_result(
                            self.env.step(actions_prb, actions_pwr),
                            local_obs_dim,
                            global_state_dim,
                        )
                        (
                            next_local_obs,
                            next_global_state,
                            reward,
                            terminated,
                            env_truncated,
                            final_info,
                        ) = transition

                        # Time-limit truncation still permits critic bootstrapping
                        # from next_global_state inside compute_gae().
                        reached_time_limit = step + 1 >= episode_length
                        truncated = bool(env_truncated or reached_time_limit)
                        buffer.push(
                            local_obs,
                            global_state,
                            actions_prb,
                            actions_pwr,
                            log_probs,
                            reward,
                            terminated,
                            truncated,
                            value,
                        )

                        local_obs = next_local_obs
                        global_state = next_global_state
                        episode_return += reward
                        global_step += 1
                        steps_taken += 1
                        if terminated or truncated:
                            break

                    if len(buffer) < 2:
                        raise RuntimeError(
                            "The current MAPPO agent requires at least two transitions "
                            "to normalize advantages without producing NaN"
                        )

                    update_metrics = self.agent.update(buffer, global_state)
                    numeric_metrics = self._numeric_metrics(update_metrics)
                    self.contract.finite_agent_parameters(self.agent)

                    record: dict[str, Any] = {
                        "episode": episode + 1,
                        "global_step": global_step,
                        "episode_return": float(episode_return),
                        "episode_length": steps_taken,
                        **self._serializable_info(final_info),
                        **numeric_metrics,
                    }
                    history.append(record)
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                    if (episode + 1) % log_interval == 0:
                        print(
                            f"episode={episode + 1} steps={global_step} "
                            f"return={episode_return:.6f} length={steps_taken}"
                        )
                    if save_interval > 0 and (episode + 1) % save_interval == 0:
                        checkpoint_manager.save(
                            checkpoint_dir / f"episode_{episode + 1:06d}.pt",
                            episode,
                            global_step,
                        )

            return history
        finally:
            if self._owns_environment and callable(getattr(self.env, "close", None)):
                self.env.close()

    def _create_environment(self) -> Any:
        """Create the integrated ISAC environment.

        Argument:
            (None).

        Return:
            (environment) (Any): Multi-BS ISAC environment instance.
        """

        try:
            from ppo.env import MultiISACEnv
        except (ImportError, ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "ppo.env or one of its physical-model dependencies is not ready. "
                "Integrate scenario, channel, sensing and communication before "
                "starting real training."
            ) from exc
        return MultiISACEnv(self.config)

    def _create_missing_agent_components(
        self, local_obs_dim: int, global_state_dim: int
    ) -> None:
        """Create the agent or rollout-buffer factory when they were not injected.

        Argument:
            (local_obs_dim) (int): Local observation dimension.
            (global_state_dim) (int): Global state dimension.

        Return:
            (None).
        """

        if self.agent is not None and self.buffer_factory is not None:
            return
        try:
            from ppo.agent import MAPPOAgent, MultiAgentRolloutBuffer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "ppo.agent is unavailable on this branch. Merge the reviewed "
                "agent implementation before starting real training."
            ) from exc

        if self.agent is None:
            self.agent = MAPPOAgent(self.config, local_obs_dim, global_state_dim)
        if self.buffer_factory is None:
            self.buffer_factory = MultiAgentRolloutBuffer

    def _reset_environment(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment and validate its return structure.

        Argument:
            (seed) (int): Episode-specific random seed.

        Return:
            (states) (tuple[np.ndarray, np.ndarray]): Local and global states.
        """

        result = self.env.reset(seed=seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "env.reset() must return (local_observations, global_state)"
            )
        return (
            np.asarray(result[0], dtype=np.float32),
            np.asarray(result[1], dtype=np.float32),
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed Python, NumPy and PyTorch random number generators.

        Argument:
            (seed) (int): Random seed shared by all training components.

        Return:
            (None).

        Note:
            Deterministic PyTorch algorithms are not forced because some CUDA
            operations may not provide deterministic implementations.
        """

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _numeric_metrics(metrics: Any) -> dict[str, float]:
        """Convert finite scalar update metrics to JSON-compatible floats.

        Argument:
            (metrics) (Any): Value returned by ``agent.update``.

        Return:
            (numeric_metrics) (dict[str, float]): Validated update metrics.
        """

        if not isinstance(metrics, Mapping):
            raise ValueError("agent.update() must return a metrics mapping")

        numeric: dict[str, float] = {}
        for key, value in metrics.items():
            value_array = np.asarray(value)
            if value_array.ndim != 0:
                raise ValueError(f"agent metric {key!s} must be scalar")
            try:
                scalar = float(value_array.item())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"agent metric {key!s} must be numeric") from exc
            if not np.isfinite(scalar):
                raise FloatingPointError(f"agent metric {key!s} is NaN or Inf")
            numeric[str(key)] = scalar
        return numeric

    @staticmethod
    def _serializable_info(
        info: Mapping[str, Any],
    ) -> dict[str, float | bool | int]:
        """Select finite scalar environment diagnostics for JSON logging.

        Argument:
            (info) (Mapping[str, Any]): Environment diagnostic mapping.

        Return:
            (scalars) (dict[str, float | bool | int]): JSON-compatible values.
        """

        scalars: dict[str, float | bool | int] = {}
        for key, value in info.items():
            if isinstance(value, (bool, np.bool_)):
                scalars[str(key)] = bool(value)
            elif isinstance(value, (int, np.integer)):
                scalars[str(key)] = int(value)
            elif isinstance(value, (float, np.floating)) and np.isfinite(value):
                scalars[str(key)] = float(value)
        return scalars


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
    return parser.parse_args()


def main() -> None:
    """Load configuration and start MAPPO training.

    Argument:
        (None).

    Return:
        (None).
    """

    args = parse_args()
    config = ProjectConfig.from_directory(args.config_dir).data
    trainer = MAPPOTrainer(
        config,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )
    trainer.run()


if __name__ == "__main__":
    main()
