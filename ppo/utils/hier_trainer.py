"""
Hierarchical ISAC Trainer (SAC at Frame level + DQN at Subframe level).

Timescale architecture:
- Outer loop (Frame level, every F subframes, e.g. F=10):
    - Agent: SACAgent (Soft Actor-Critic)
    - State: s_SAC(f) via env.get_sac_state()
    - Action: a_SAC(f) [B, W, M+S] macro PRB allocation ratios per target/slice
    - Budget: env.begin_frame(sac_ratios, num_subframes=F) sets integer PRB caps
    - Reward: r_SAC(f) via frame_summary["reward"]
- Inner loop (Subframe level, t = 1..F):
    - Agent: DQNAgent (Deep Q-Network)
    - State: s_b^DQN(t) via env.get_dqn_state() or global_state
    - Action: a_b^DQN(t) [B, sum(nPrb)] discrete PRB allocation choices in {0..M+K}
    - Reward: r_DQN(t) from env.step() team reward
"""

from __future__ import annotations

import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

# Ensure project root is on sys.path for direct script invocation
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from ppo.utils.checkpoint_manager import CheckpointManager
from ppo.utils.project_config import ProjectConfig
from ppo.utils.result_artifacts import ResultArtifacts
from ppo.utils.runtime_contract import RuntimeContract


def load_hierarchical_config(config_dir: str | Path) -> dict[str, Any]:
    """Load and validate hierarchical configuration via ProjectConfig."""
    return ProjectConfig.from_directory(config_dir, mode="hierarchical").data


class HierarchicalISACTrainer:
    """Coordinate 2-timescale Hierarchical RL training for Multi-BWP ISAC."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        env: Any | None = None,
        sac_agent: Any | None = None,
        dqn_agent: Any | None = None,
        output_dir: str | Path | None = "results",
        resume_from: str | Path | None = None,
        subframes_per_frame: int | None = None,
        single_bwp_mode: bool = False,
    ):
        """Initialize Hierarchical ISAC Trainer with strict config & runtime contracts.

        Args:
            config: Merged project configuration dictionary.
            env: Optional pre-instantiated HierISACEnv.
            sac_agent: Optional pre-instantiated SACAgent.
            dqn_agent: Optional pre-instantiated DQNAgent.
            output_dir: Root directory for metrics, figures, and checkpoints.
            resume_from: Optional path to an existing checkpoint to resume from.
            subframes_per_frame: Optional override for F (subframes per frame).
            single_bwp_mode: If True, enforce single-BWP allocation constraint.
        """
        project_config = ProjectConfig(config, mode="hierarchical")
        project_config.validate()
        self.config = project_config.data
        self.contract = RuntimeContract(self.config)

        self.env = env
        self.sac_agent = sac_agent
        self.dqn_agent = dqn_agent
        self._owns_environment = env is None

        self.single_bwp_mode = bool(single_bwp_mode)
        self.model_name = (
            "hierarchical_single_bwp" if self.single_bwp_mode else "hierarchical"
        )

        project_root = Path(__file__).resolve().parents[2]
        self._project_root = project_root
        resolved_out = "results" if output_dir is None else output_dir
        out_path = Path(resolved_out)
        self.output_dir = (
            out_path if out_path.is_absolute() else (project_root / out_path)
        )

        self.resume_from = Path(resume_from) if resume_from is not None else None

        hier_cfg = self.config["hierarchical_parameters"]
        time_cfg = hier_cfg["timescale"]
        train_cfg = hier_cfg["training"] if "training" in hier_cfg else time_cfg
        sac_cfg = hier_cfg["sac"]
        dqn_cfg = hier_cfg["dqn"]

        if subframes_per_frame is not None:
            if (
                isinstance(subframes_per_frame, bool)
                or not isinstance(subframes_per_frame, int)
                or subframes_per_frame <= 0
            ):
                raise ValueError("subframes_per_frame must be a positive integer")
            self.subframes_per_frame = int(subframes_per_frame)
        else:
            self.subframes_per_frame = int(time_cfg["subframes_per_frame"])

        self.total_episodes = int(train_cfg["total_episodes"])
        self.episode_length = int(train_cfg["episode_length"])
        self.save_interval = int(train_cfg["save_interval"])
        self.log_interval = int(train_cfg["log_interval"])
        self.seed = int(train_cfg["seed"])

        device_str = str(train_cfg["device"])
        self.device = torch.device(
            device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        sys_config = self.config["system_parameters"]
        self.num_bs = int(sys_config["num_bs"])
        self.num_targets = int(sys_config["num_target"])

        num_embb = int(sys_config["num_embb"])
        num_urllc = int(sys_config["num_urllc"])
        num_ue_1embb = int(sys_config["num_ue_1embb"])
        num_ue_1urllc = int(sys_config["num_ue_1urllc"])
        self.num_ues = num_embb * num_ue_1embb + num_urllc * num_ue_1urllc
        self.n_prb = int(sum(sys_config["nPrb"]))
        self.num_slices = num_embb + num_urllc + self.num_targets

        self._bwp = list(sys_config["bwp"])
        self.num_bwp_ru = [len(self._bwp)] * self.num_bs

        self.sac_batch_size = int(sac_cfg["batch_size"])
        self.sac_policy_delay = int(sac_cfg["policy_delay"])
        self.dqn_batch_size = int(dqn_cfg["batch_size"])

    def _create_environment(self) -> Any:
        """Instantiate HierISACEnv in multi-BWP or single-BWP mode."""
        try:
            from ppo.envs.HIERenv import HierISACEnv
        except (ImportError, ModuleNotFoundError, SyntaxError) as exc:
            raise RuntimeError(
                "ppo.envs.HIERenv is unavailable or failed to import."
            ) from exc
        return HierISACEnv(
            self.config,
            cost_max=None,
            bwp_mode=not self.single_bwp_mode,
        )

    def _create_missing_agent_components(
        self, sac_state_dim: int, dqn_state_dim: int
    ) -> None:
        """Create SAC and DQN agents when they were not injected."""
        hier_cfg = self.config["hierarchical_parameters"]
        sac_cfg = hier_cfg["sac"]
        dqn_cfg = hier_cfg["dqn"]

        if self.sac_agent is None:
            try:
                from ppo.agents.SACagent import SACAgent
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "ppo.agents.SACagent is unavailable; cannot create SACAgent."
                ) from exc

            actor_lr = float(sac_cfg["actor_lr"])
            critic_lr = float(sac_cfg["critic_lr"])
            alpha_lr = float(sac_cfg["alpha_lr"])
            sac_train_cons = {
                "device": self.device,
                "gamma": float(sac_cfg["gamma"]),
                "tau": float(sac_cfg["tau"]),
                "alpha": float(sac_cfg["init_alpha"]),
                "action_scale": float(sac_cfg["action_scale"]),
                "action_bias": float(sac_cfg["action_bias"]),
                "lr": actor_lr,
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "alpha_lr": alpha_lr,
            }
            self.sac_agent = SACAgent(
                state_dim=sac_state_dim,
                num_rus=self.num_bs,
                num_bwp_ru=self.num_bwp_ru,
                num_slices=self.num_slices,
                train_cons=sac_train_cons,
            )

        if self.dqn_agent is None:
            try:
                from ppo.agents.DQNAgent import DQNAgent
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "ppo.agents.DQNAgent is unavailable. Provide a valid DQNAgent "
                    "implementation or inject dqn_agent before running training."
                ) from exc

            self.dqn_agent = DQNAgent(
                state_dim=dqn_state_dim,
                n_prb=self.n_prb,
                num_targets=self.num_targets,
                num_ues=self.num_ues,
                lr=float(dqn_cfg["lr"]),
                gamma=float(dqn_cfg["gamma"]),
                epsilon_start=float(dqn_cfg["epsilon_start"]),
                epsilon_min=float(dqn_cfg["epsilon_min"]),
                epsilon_decay=float(dqn_cfg["epsilon_decay"]),
                target_update_freq=int(dqn_cfg["target_update_freq"]),
                device=self.device,
            )

    def _reset_environment(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment and validate its return structure."""
        result = self.env.reset(seed=seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "env.reset() must return (local_observations, global_state)"
            )
        return (
            np.asarray(result[0], dtype=np.float32),
            np.asarray(result[1], dtype=np.float32),
        )

    def _get_dqn_state(self, fallback_global_state: np.ndarray) -> np.ndarray:
        """Read and validate finite subframe-level DQN state."""
        if hasattr(self.env, "get_dqn_state"):
            raw_state = self.env.get_dqn_state()
        else:
            raw_state = fallback_global_state
        state = np.asarray(raw_state, dtype=np.float32)
        if state.ndim not in (1, 2) or state.size == 0:
            raise ValueError("DQN state must be a non-empty 1D or 2D array")
        if not np.all(np.isfinite(state)):
            raise ValueError("DQN state must contain only finite values")
        return state

    def _select_dqn_actions(self, dqn_state: np.ndarray) -> np.ndarray:
        """Select and contract-validate discrete PRB actions from DQNAgent."""
        selector = (
            self.dqn_agent.select_actions_all_bs
            if callable(getattr(self.dqn_agent, "select_actions_all_bs", None))
            else self.dqn_agent.select_action
        )
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(selector).parameters
        except (TypeError, ValueError):
            params = {}
        if "budget" in params:
            kwargs["budget"] = getattr(self.env, "budget", None)
        if "bwp_mode" in params:
            kwargs["bwp_mode"] = not self.single_bwp_mode
        raw_actions = selector(dqn_state, **kwargs)
        return self.contract.dqn_actions(raw_actions)

    def _enforce_single_bwp(self, a_sac: np.ndarray) -> np.ndarray:
        """Enforce Benchmark: Hierarchical-singleBWP constraint .

        For each RU b and slice s, keep only the BWP with maximum allocation
        and zero out all other BWPs.
        """
        arr = np.asarray(a_sac, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(self.num_bs, -1, self.num_slices)
        elif arr.ndim == 2 and self.num_bs == 1:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        a_single = np.zeros_like(arr)
        for b in range(self.num_bs):
            for s in range(self.num_slices):
                best_w = int(np.argmax(arr[b, :, s]))
                a_single[b, best_w, s] = arr[b, best_w, s]
        return a_single

    def run(
        self,
        total_episodes: int | None = None,
        episode_length: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run 2-timescale hierarchical training across episodes.

        Args:
            total_episodes: Optional override for total episodes.
            episode_length: Optional override for subframes per episode.

        Returns:
            history: List of per-episode metric dictionaries recorded in this run.
        """
        if total_episodes is not None:
            if (
                isinstance(total_episodes, bool)
                or not isinstance(total_episodes, int)
                or total_episodes <= 0
            ):
                raise ValueError("total_episodes must be a positive integer")
            episodes_to_run = int(total_episodes)
        else:
            episodes_to_run = self.total_episodes

        if episode_length is not None:
            if (
                isinstance(episode_length, bool)
                or not isinstance(episode_length, int)
                or episode_length <= 0
            ):
                raise ValueError("episode_length must be a positive integer")
            steps_per_episode = int(episode_length)
        else:
            steps_per_episode = self.episode_length

        self._seed_everything(self.seed)
        self.env = self.env if self.env is not None else self._create_environment()

        try:
            initial_local_obs, initial_global_state = self._reset_environment(self.seed)
            local_obs_dim, global_state_dim = self.contract.observation_dimensions(
                initial_local_obs, initial_global_state
            )
            _, sac_state_dim = self.contract.sac_state(self.env.get_sac_state())
            initial_dqn_state = self._get_dqn_state(initial_global_state)
            dqn_state_dim = int(
                getattr(self.env, "dqn_state_dim", initial_dqn_state.shape[-1])
            )

            self._create_missing_agent_components(sac_state_dim, dqn_state_dim)
            self.contract.sac_agent_structure(self.sac_agent, sac_state_dim)
            self.contract.dqn_agent_structure(self.dqn_agent, dqn_state_dim)
            self.contract.finite_hier_parameters(self.sac_agent, self.dqn_agent)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifacts = ResultArtifacts(self.output_dir, run_name=self.model_name)
            metrics_path = artifacts.prepare_metrics(resume=self.resume_from is not None)
            checkpoint_dir = self.output_dir / "checkpoints" / self.model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_manager = CheckpointManager(
                self.config,
                sac_agent=self.sac_agent,
                dqn_agent=self.dqn_agent,
                mode="hierarchical",
            )

            start_episode = 0
            global_step = 0
            if self.resume_from is not None:
                resume_candidates = (
                    self.resume_from,
                    self._project_root / self.resume_from,
                    checkpoint_dir / self.resume_from.name,
                )
                resume_target = next(
                    (candidate for candidate in resume_candidates if candidate.is_file()),
                    self.resume_from,
                )
                start_episode, global_step = checkpoint_manager.load(resume_target)
                self.contract.finite_hier_parameters(self.sac_agent, self.dqn_agent)

            if start_episode > episodes_to_run:
                raise ValueError(
                    "Checkpoint episode exceeds hierarchical_parameters total_episodes"
                )

            file_mode = checkpoint_manager.prepare_metrics(metrics_path, start_episode)
            history: list[dict[str, Any]] = []
            last_actor_loss = torch.tensor(0.0, device=self.device)
            sac_step = global_step // max(1, self.subframes_per_frame)
            expected_sac_action_dim = getattr(
                self.sac_agent,
                "action_dim",
                sum(self.num_bwp_ru) * self.num_slices,
            )

            with metrics_path.open(file_mode, encoding="utf-8") as metrics_file:
                for episode in range(start_episode, episodes_to_run):
                    local_obs, global_state = self._reset_environment(
                        self.seed + episode
                    )
                    reset_dims = self.contract.observation_dimensions(
                        local_obs, global_state
                    )
                    if reset_dims != (local_obs_dim, global_state_dim):
                        raise ValueError(
                            "observation dimensions changed between episodes: "
                            f"expected {(local_obs_dim, global_state_dim)}, "
                            f"received {reset_dims}"
                        )

                    s_sac, ep_sac_dim = self.contract.sac_state(
                        self.env.get_sac_state()
                    )
                    if ep_sac_dim != sac_state_dim:
                        raise ValueError(
                            "SAC state dimension changed between episodes: "
                            f"expected {sac_state_dim}, got {ep_sac_dim}"
                        )

                    episode_return = 0.0
                    ep_sac_rewards: list[float] = []
                    ep_dqn_rewards: list[float] = []
                    ep_dqn_losses: list[float] = []
                    ep_sac_actor_losses: list[float] = []
                    ep_sac_critic_losses: list[float] = []
                    final_info: dict[str, Any] = {}
                    info_sums: dict[str, float] = {}
                    info_counts: dict[str, int] = {}
                    subframe_count = 0

                    while subframe_count < steps_per_episode:
                        # 1. Macro-timescale: SAC selects BWP PRB budget
                        s_sac, step_sac_dim = self.contract.sac_state(s_sac)
                        if step_sac_dim != sac_state_dim:
                            raise ValueError(
                                "SAC state dimension changed during episode: "
                                f"expected {sac_state_dim}, got {step_sac_dim}"
                            )

                        a_sac = np.asarray(
                            self.sac_agent.select_action(s_sac), dtype=np.float32
                        )
                        flat_sac_action = self.contract.sac_action(
                            a_sac.reshape(-1), expected_dim=expected_sac_action_dim
                        )
                        if self.single_bwp_mode:
                            a_sac = self._enforce_single_bwp(flat_sac_action)
                            flat_sac_action = self.contract.sac_action(
                                a_sac.reshape(-1), expected_dim=expected_sac_action_dim
                            )

                        frames_to_run = min(
                            self.subframes_per_frame,
                            steps_per_episode - subframe_count,
                        )

                        if hasattr(self.env, "begin_frame"):
                            sac_ratios = flat_sac_action.reshape(
                                self.num_bs, len(self._bwp), self.num_slices
                            )
                            self.env.begin_frame(sac_ratios, num_subframes=frames_to_run)

                        terminated = False
                        truncated = False
                        frames_ran = 0

                        # 2. Micro-timescale: DQN runs subframes
                        for _ in range(frames_to_run):
                            s_dqn = self._get_dqn_state(global_state)
                            actions_prb = self._select_dqn_actions(s_dqn)

                            step_out = self.env.step(actions_prb)
                            transition = self.contract.hier_step_result(
                                step_out, local_obs_dim, global_state_dim
                            )
                            (
                                next_local_obs,
                                next_global_state,
                                env_reward,
                                terminated,
                                env_truncated,
                                final_info,
                            ) = transition
                            self._accumulate_float_info(
                                final_info, info_sums, info_counts
                            )

                            r_dqn = float(env_reward)
                            s_dqn_next = self._get_dqn_state(next_global_state)

                            for b in range(self.num_bs):
                                self.dqn_agent.replay_buffer.push(
                                    s_dqn[b] if s_dqn.ndim > 1 else s_dqn,
                                    actions_prb[b],
                                    r_dqn,
                                    s_dqn_next[b] if s_dqn_next.ndim > 1 else s_dqn_next,
                                    float(terminated),
                                )

                            dqn_update_out = self.dqn_agent.update(
                                batch_size=self.dqn_batch_size
                            )
                            if isinstance(dqn_update_out, Mapping):
                                dqn_loss = float(
                                    dqn_update_out.get(
                                        "dqn_loss", dqn_update_out.get("loss", 0.0)
                                    )
                                )
                            else:
                                dqn_loss = float(dqn_update_out)
                            if not np.isfinite(dqn_loss):
                                raise FloatingPointError(
                                    "DQN update returned non-finite loss"
                                )
                            if dqn_loss > 0:
                                ep_dqn_losses.append(dqn_loss)
                            self.contract.finite_hier_parameters(
                                self.sac_agent, self.dqn_agent
                            )

                            ep_dqn_rewards.append(r_dqn)
                            episode_return += env_reward
                            local_obs = next_local_obs
                            global_state = next_global_state
                            global_step += 1
                            subframe_count += 1
                            frames_ran += 1

                            reached_time_limit = subframe_count >= steps_per_episode
                            truncated = bool(env_truncated or reached_time_limit)
                            if terminated or truncated:
                                break

                        # 3. End of Frame: SAC Reward and Update
                        s_sac_next, next_sac_dim = self.contract.sac_state(
                            self.env.get_sac_state()
                        )
                        if next_sac_dim != sac_state_dim:
                            raise ValueError(
                                "SAC state dimension changed during episode: "
                                f"expected {sac_state_dim}, got {next_sac_dim}"
                            )

                        if isinstance(final_info.get("frame_summary"), Mapping):
                            r_sac = float(final_info["frame_summary"]["reward"])
                        elif getattr(self.env, "last_frame_summary", None) is not None:
                            r_sac = float(self.env.last_frame_summary["reward"])
                        else:
                            r_sac = float(np.mean(ep_dqn_rewards[-frames_ran:]))
                        if not np.isfinite(r_sac):
                            raise ValueError("SAC reward must be finite")

                        ep_sac_rewards.append(r_sac)
                        self.sac_agent.replay_buffer.push(
                            s_sac,
                            flat_sac_action,
                            r_sac,
                            s_sac_next,
                            float(terminated),
                        )

                        sac_step += 1
                        if len(self.sac_agent.replay_buffer) >= self.sac_batch_size:
                            actor_loss, critic_loss, last_actor_loss = (
                                self.sac_agent.update(
                                    step=sac_step,
                                    policy_delay=self.sac_policy_delay,
                                    last_actor_loss=last_actor_loss,
                                    batch_size=self.sac_batch_size,
                                )
                            )
                            actor_loss_val = float(
                                actor_loss.item()
                                if isinstance(actor_loss, torch.Tensor)
                                else actor_loss
                            )
                            critic_loss_val = float(
                                critic_loss.item()
                                if isinstance(critic_loss, torch.Tensor)
                                else critic_loss
                            )
                            if not (
                                np.isfinite(actor_loss_val)
                                and np.isfinite(critic_loss_val)
                            ):
                                raise FloatingPointError(
                                    "SAC update returned non-finite loss"
                                )
                            ep_sac_actor_losses.append(actor_loss_val)
                            ep_sac_critic_losses.append(critic_loss_val)
                            self.contract.finite_hier_parameters(
                                self.sac_agent, self.dqn_agent
                            )

                        s_sac = s_sac_next
                        if terminated or truncated:
                            break

                    mean_sac_r = (
                        float(np.mean(ep_sac_rewards)) if ep_sac_rewards else 0.0
                    )
                    mean_dqn_r = (
                        float(np.mean(ep_dqn_rewards)) if ep_dqn_rewards else 0.0
                    )
                    mean_dqn_loss = (
                        float(np.mean(ep_dqn_losses)) if ep_dqn_losses else 0.0
                    )
                    mean_sac_actor_loss = (
                        float(np.mean(ep_sac_actor_losses))
                        if ep_sac_actor_losses
                        else 0.0
                    )
                    mean_sac_critic_loss = (
                        float(np.mean(ep_sac_critic_losses))
                        if ep_sac_critic_losses
                        else 0.0
                    )
                    dqn_eps = float(getattr(self.dqn_agent, "epsilon", 0.0))
                    sac_alpha = float(getattr(self.sac_agent, "alpha", 0.0))

                    record: dict[str, Any] = {
                        **self._serializable_info(final_info),
                        **self._mean_float_info(info_sums, info_counts),
                        "episode": episode + 1,
                        "global_step": global_step,
                        "episode_return": float(episode_return),
                        "episode_length": subframe_count,
                        "mean_sac_reward": mean_sac_r,
                        "mean_dqn_reward": mean_dqn_r,
                        "sac_actor_loss": mean_sac_actor_loss,
                        "sac_critic_loss": mean_sac_critic_loss,
                        "sac_alpha": sac_alpha,
                        "dqn_loss": mean_dqn_loss,
                        "dqn_epsilon": dqn_eps,
                    }
                    history.append(record)
                    metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    metrics_file.flush()

                    if (episode + 1) % self.log_interval == 0:
                        print(
                            f"episode={episode + 1}/{episodes_to_run} "
                            f"steps={global_step} return={episode_return:.6f} "
                            f"length={subframe_count} sac_r={mean_sac_r:.3f} "
                            f"dqn_r={mean_dqn_r:.3f} dqn_loss={mean_dqn_loss:.4f} "
                            f"eps={dqn_eps:.3f}"
                        )

                    if (
                        self.save_interval > 0
                        and (episode + 1) % self.save_interval == 0
                    ) or episode + 1 == episodes_to_run:
                        checkpoint_manager.save(
                            checkpoint_dir / f"episode_{episode + 1:06d}.pt",
                            episode,
                            global_step,
                        )

            figure_paths = artifacts.render(metrics_path)
            print(
                f"figures saved to {artifacts.figures_dir} "
                f"({len(figure_paths)} PNG files)"
            )
            if hasattr(self.env, "plot_sensing_resource_usage"):
                try:
                    plot_fn = self.env.plot_sensing_resource_usage
                    try:
                        plot_params = inspect.signature(plot_fn).parameters
                    except (TypeError, ValueError):
                        plot_params = {}
                    accepts_save_dir = "save_dir" in plot_params or any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in plot_params.values()
                    )
                    if accepts_save_dir:
                        plot_fn(save_dir=str(artifacts.figures_dir))
                    else:
                        plot_fn()
                except Exception as exc:
                    print(f"Warning: could not plot sensing resource usage: {exc}")
            return history
        finally:
            if self._owns_environment and callable(getattr(self.env, "close", None)):
                self.env.close()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed Python, NumPy, and PyTorch random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _serializable_info(
        info: Mapping[str, Any],
    ) -> dict[str, float | bool | int]:
        """Select finite scalar environment diagnostics for JSON logging."""
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
        """Accumulate finite floating-point environment metrics."""
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
        """Compute episode means for accumulated environment metrics."""
        return {
            name: total / counts[name]
            for name, total in sums.items()
            if counts.get(name, 0) > 0
        }


if __name__ == "__main__":
    from ppo.train_hierarchical import main

    main()
