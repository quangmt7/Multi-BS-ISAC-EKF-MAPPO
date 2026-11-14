"""Coordinate episodic on-policy MAPPO training."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ppo.envs.env import BaseISACEnv
from ppo.utils.checkpoint_manager import CheckpointManager
from ppo.utils.project_config import ProjectConfig
from ppo.utils.result_artifacts import ResultArtifacts
from ppo.utils.runtime_contract import RuntimeContract


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
        single_bwp_mode: bool = False,
    ):
        """Initialize dependencies without starting training.

        Argument:
            (config) (Mapping[str, Any]): Merged project configuration.
            (env) (Any | None): Optional injected environment.
            (agent) (Any | None): Optional injected agent.
            (buffer_factory) (Callable[[], Any] | None): Rollout-buffer constructor.
            (output_dir) (str | Path): Result directory.
            (resume_from) (str | Path | None): Optional checkpoint path.
            (single_bwp_mode) (bool): Restrict each slice/target to at most one BWP.

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
        self.single_bwp_mode = bool(single_bwp_mode)
        self.model_name = (
            "mappo_single_bwp" if self.single_bwp_mode else "mappo"
        )
        self._project_root = Path(__file__).resolve().parents[2]
        out_path = Path(output_dir)
        self.output_dir = (
            out_path if out_path.is_absolute() else (self._project_root / out_path)
        )
        if resume_from is not None:
            res_path = Path(resume_from)
            self._raw_resume_from: Path | None = res_path
            self.resume_from: Path | None = (
                res_path
                if (res_path.is_absolute() or res_path.is_file())
                else (self._project_root / res_path)
            )
        else:
            self._raw_resume_from = None
            self.resume_from = None
        self._owns_environment = env is None

    def run(
        self,
        bwp_mode: bool | None = None,
        single_bwp_mode: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Run training and return metrics produced in this process.

        Argument:
            (bwp_mode) (bool | None): Legacy flag. Whether to allow multi-BWP PRB
                allocation (``True``) or restrict each slice/target to at most one
                BWP (``False``).
            (single_bwp_mode) (bool | None): Restrict each slice/target per BS
                to at most one BWP (``True``) or allow multi-BWP allocation
                (``False``).

        Return:
            (history) (list[dict[str, Any]]): Episode metrics from this run.
        """

        if single_bwp_mode is not None:
            self.single_bwp_mode = bool(single_bwp_mode)
        elif bwp_mode is not None:
            self.single_bwp_mode = not bool(bwp_mode)

        self.model_name = (
            "mappo_single_bwp" if self.single_bwp_mode else "mappo"
        )
        effective_bwp_mode = not self.single_bwp_mode

        training = self.config["mappo_parameters"]["training"]
        ppo = self.config["mappo_parameters"]["ppo"]
        total_episodes = int(training["total_episodes"])
        episode_length = int(training["episode_length"])
        save_interval = int(training["save_interval"])
        log_interval = int(training["log_interval"])
        update_interval = int(training["update_interval"])
        seed = int(training["seed"])
        use_linear_lr_decay = bool(ppo["use_linear_lr_decay"])

        self._seed_everything(seed)
        self.env = (
            self.env
            if self.env is not None
            else self._create_environment(effective_bwp_mode)
        )

        try:
            initial_local_obs, initial_global_state = self._reset_environment(seed)
            local_obs_dim, global_state_dim = self.contract.observation_dimensions(
                initial_local_obs, initial_global_state
            )
            self._create_missing_agent_components(local_obs_dim, global_state_dim)
            # Policy log-probabilities must cover only actions applied by the environment.
            if isinstance(self.env, BaseISACEnv) and hasattr(getattr(self.agent, "actornetwork", None), "power_mean"):
                raise RuntimeError(
                    "The Env uses fixed pEach power. Migrate the PPO Actor/log-probability "
                    "and rollout contract to PRB-only actions before training (state/trainer task)."
                )
            self.contract.agent_structure(self.agent, local_obs_dim, global_state_dim)
            self.contract.finite_agent_parameters(self.agent)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifacts = ResultArtifacts(self.output_dir, run_name=self.model_name)
            metrics_path = artifacts.prepare_metrics(resume=self.resume_from is not None)
            checkpoint_dir = self.output_dir / "checkpoints" / self.model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_manager = CheckpointManager(self.config, ppo_agent=self.agent, mode="mappo")

            start_episode = 0
            global_step = 0
            if self.resume_from is not None:
                resume_target = self._resolve_resume_target(checkpoint_dir)
                start_episode, global_step = checkpoint_manager.load(resume_target)
            if start_episode > total_episodes:
                raise ValueError(
                    "Checkpoint episode exceeds mappo_parameters.training.total_episodes"
                )

            file_mode = checkpoint_manager.prepare_metrics(metrics_path, start_episode)
            history: list[dict[str, Any]] = []
            buffer = self.buffer_factory()

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

                    episode_return = 0.0
                    final_info: dict[str, Any] = {}
                    info_sums: dict[str, float] = {}
                    info_counts: dict[str, int] = {}
                    steps_taken = 0
                    accepts_power = self._env_accepts_power()

                    for step in range(episode_length):
                        policy_output = self.contract.policy_output(
                            self.agent.select_action(local_obs, global_state)
                        )
                        actions_prb, actions_pwr, log_probs, value = policy_output
                        step_out = (
                            self.env.step(actions_prb, actions_pwr)
                            if accepts_power
                            else self.env.step(actions_prb)
                        )
                        transition = self.contract.step_result(
                            step_out,
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
                        self._accumulate_float_info(final_info, info_sums, info_counts)

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
                            self.agent.estimate_value(next_global_state),
                        )

                        local_obs = next_local_obs
                        global_state = next_global_state
                        episode_return += reward
                        global_step += 1
                        steps_taken += 1
                        if terminated or truncated:
                            break
                            

                    # Keep on-policy transitions until the scheduled update.
                    numeric_metrics = {}
                    should_update = (update_interval > 0 and
                                     ((episode + 1) % update_interval == 0 or
                                      episode + 1 == total_episodes))
                    if should_update and len(buffer) > 0:
                        update_metrics = self.agent.update(buffer)
                        numeric_metrics = self._numeric_metrics(update_metrics)
                        self.contract.finite_agent_parameters(self.agent)
                        if use_linear_lr_decay and callable(
                            getattr(self.agent, "update_learning_rate", None)
                        ):
                            self.agent.update_learning_rate(episode, total_episodes)

                    record: dict[str, Any] = {
                        **self._serializable_info(final_info),
                        **self._mean_float_info(info_sums, info_counts),
                        **numeric_metrics,
                        "episode": episode + 1,
                        "global_step": global_step,
                        "episode_return": float(episode_return),
                        "episode_length": steps_taken,
                    }
                    history.append(record)
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                    if (episode + 1) % log_interval == 0:
                        print(
                            f"episode={episode + 1} steps={global_step} "
                            f"return={episode_return:.6f} length={steps_taken}"
                        )
                    if (should_update and
                            ((save_interval > 0 and (episode + 1) % save_interval == 0)
                             or episode + 1 == total_episodes)):
                        checkpoint_manager.save(
                            checkpoint_dir / f"episode_{episode + 1:06d}.pt",
                            episode,
                            global_step,
                        )

            figure_paths = artifacts.render(metrics_path)
            print(f"figures saved to {artifacts.figures_dir} ({len(figure_paths)} PNG files)")
            if callable(getattr(self.env, "plot_sensing_resource_usage", None)):
                try:
                    self._call_plot_sensing_resource_usage(artifacts.figures_dir)
                except Exception as exc:
                    print(f"Warning: could not plot sensing resource usage: {exc}")
            return history
        finally:
            if self._owns_environment and callable(getattr(self.env, "close", None)):
                self.env.close()

    def _resolve_resume_target(self, checkpoint_dir: Path) -> Path:
        """Resolve an existing checkpoint path from cwd, project root, or checkpoint_dir."""
        raw_path = self._raw_resume_from
        if raw_path is None:
            raise ValueError("resume_from is not configured")
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(self._project_root / raw_path)
        candidates.append(checkpoint_dir / raw_path.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return self.resume_from if self.resume_from is not None else raw_path

    def _call_plot_sensing_resource_usage(self, figures_dir: Path) -> None:
        """Call env.plot_sensing_resource_usage with save_dir when accepted."""
        plot_fn = self.env.plot_sensing_resource_usage
        try:
            params = inspect.signature(plot_fn).parameters
        except (TypeError, ValueError):
            params = {}
        accepts_save_dir = "save_dir" in params or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )
        if accepts_save_dir:
            plot_fn(save_dir=str(figures_dir))
        else:
            plot_fn()

    def _env_accepts_power(self) -> bool:
        """Return whether an injected non-BaseISACEnv accepts (prb, pwr) in step()."""
        if isinstance(self.env, BaseISACEnv):
            return False
        try:
            params = list(inspect.signature(self.env.step).parameters.values())
        except (TypeError, ValueError):
            return False
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return True
        positional = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 2

    def _create_environment(self, bwp_mode=True) -> Any:
        """Create the integrated ISAC environment.

        Argument:
            (None).

        Return:
            (environment) (Any): Multi-BS ISAC environment instance.
        """

        try:
            from ppo.envs.PPOenv import PPOISACEnv
        except (ImportError, ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "ppo.env or one of its physical-model dependencies is not ready. "
                "Integrate scenario, channel, sensing and communication before "
                "starting real training."
            ) from exc
        return PPOISACEnv(self.config, bwp_mode=bwp_mode)

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
            from ppo.agents.PPOagent import MAPPOAgent
            from ppo.agents.AgentComponents import MultiAgentRolloutBuffer
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

    @staticmethod
    def _accumulate_float_info(
        info: Mapping[str, Any],
        sums: dict[str, float],
        counts: dict[str, int],
    ) -> None:
        """Accumulate finite floating-point environment metrics.

        Argument:
            (info) (Mapping[str, Any]): Diagnostics from one environment step.
            (sums) (dict[str, float]): Running sum for each floating metric.
            (counts) (dict[str, int]): Number of samples for each floating metric.

        Return:
            (None).

        Note:
            Boolean flags and integer counters remain final-step values. Floating
            metrics such as rate, sensing probability, penalty and interference
            are averaged over the episode.
        """

        for key, value in info.items():
            if not isinstance(value, (float, np.floating)) or not np.isfinite(value):
                continue
            name = str(key)
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1

    @staticmethod
    def _mean_float_info(
        sums: Mapping[str, float], counts: Mapping[str, int]
    ) -> dict[str, float]:
        """Compute episode means for accumulated environment metrics.

        Argument:
            (sums) (Mapping[str, float]): Sum of each floating metric.
            (counts) (Mapping[str, int]): Number of samples for each metric.

        Return:
            (means) (dict[str, float]): Per-episode mean metrics.
        """

        return {
            name: total / counts[name]
            for name, total in sums.items()
            if counts.get(name, 0) > 0
        }
