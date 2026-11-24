"""SAC frame controller and quota-constrained DQN subframe training."""

from pathlib import Path
from typing import Any, Mapping
import json
import random
import numpy as np
import torch
from ppo.agents.allocation import sac_ratios
from ppo.agents.SACagent import SACAgent
from ppo.agents.DQNagent import DQNAgent
from ppo.envs.HIERenv import HierISACEnv
from ppo.utils.project_config import ProjectConfig, episode_scenario_seed
from ppo.utils.runtime_contract import RuntimeContract
from ppo.utils.checkpoint_manager import CheckpointManager
from ppo.utils.result_artifacts import ResultArtifacts


def load_hierarchical_config(config_dir):
    return ProjectConfig.from_directory(config_dir, mode="hierarchical").data


class HierarchicalISACTrainer:
    def __init__(
        self,
        config,
        *,
        env=None,
        sac_agent=None,
        dqn_agent=None,
        output_dir="results",
        resume_from=None,
        subframes_per_frame=None,
        single_bwp_mode=False,
    ):
        project = ProjectConfig(config, mode="hierarchical")
        project.validate()
        self.config = project.data
        self.contract = RuntimeContract(self.config)
        self.env, self.sac_agent, self.dqn_agent = env, sac_agent, dqn_agent
        self.single_bwp_mode = bool(single_bwp_mode)
        self.model_name = (
            "hierarchical_single_bwp" if single_bwp_mode else "hierarchical"
        )
        root = Path(__file__).resolve().parents[2]
        out = Path("results" if output_dir is None else output_dir)
        self.output_dir = out if out.is_absolute() else root / out
        resume = Path(resume_from) if resume_from is not None else None
        self.resume_from = (
            resume
            if resume is None or resume.is_absolute() or resume.is_file()
            else root / resume
        )
        hier = self.config["hierarchical_parameters"]
        self.dqn_warmup_steps = hier["dqn"].get("warmup_steps", 0)
        self.epsilon_decay_unit = hier["dqn"].get("epsilon_decay_unit", "subframe")
        self.train_cfg = hier.get("training", hier["timescale"])
        self.subframes_per_frame = (
            hier["timescale"]["subframes_per_frame"]
            if subframes_per_frame is None
            else subframes_per_frame
        )
        if (
            isinstance(self.subframes_per_frame, bool)
            or not isinstance(self.subframes_per_frame, int)
            or self.subframes_per_frame <= 0
        ):
            raise ValueError("subframes_per_frame must be a positive integer")
        device = self.train_cfg["device"]
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.seed = self.train_cfg["seed"]

    def _create_missing_agent_components(self, sac_state_dim, dqn_state_dim):
        if self.sac_agent is None:
            self.sac_agent = SACAgent(self.config, sac_state_dim, self.device)
        if self.dqn_agent is None:
            self.dqn_agent = DQNAgent(
                self.config, dqn_state_dim, self.device, single_bwp=self.single_bwp_mode
            )

    def _store_dqn(self, pending, next_state, next_budget, global_step, losses):
        state, action, reward, terminated, budget = pending
        for b in range(state.shape[0]):
            self.dqn_agent.replay_buffer.push(
                state[b],
                action[b],
                reward,
                next_state[b],
                terminated,
                budget[b],
                next_budget[b],
            )
        if global_step >= self.dqn_warmup_steps:
            metrics = self.dqn_agent.update()
            if metrics["updated"]:
                losses.append(metrics["dqn_loss"])

    def _bootstrap_dqn(self, pending, sac_next, global_step, losses):
        """Build a hypothetical continuation at an external episode timeout."""
        if pending[3]:
            full_caps = np.broadcast_to(
                np.asarray(self.env.nPrb)[None, :, None], self.env.BWP_slice.shape
            )
            self._store_dqn(
                pending, self.env.get_dqn_state(), full_caps, global_step, losses
            )
            return
        raw = self.sac_agent.select_action(sac_next)
        ratios = sac_ratios(raw, self.env.BWP_slice.shape, self.single_bwp_mode)
        next_budget, _ = self.env.apportion_budget(ratios)
        saved_budget = self.env.budget
        try:
            self.env.set_budget(next_budget)
            next_state = self.env.get_dqn_state()
        finally:
            self.env.set_budget(saved_budget)
        self._store_dqn(pending, next_state, next_budget, global_step, losses)

    def run(self, total_episodes=None, episode_length=None):
        episodes = (
            self.train_cfg["total_episodes"]
            if total_episodes is None
            else total_episodes
        )
        steps = (
            self.train_cfg["episode_length"]
            if episode_length is None
            else episode_length
        )
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (episodes, steps)
        ):
            raise ValueError("Episode count and length must be positive integers")
        self.config["system_parameters"]["simulation"]["num_slots_per_episode"] = steps
        self.config["system_parameters"]["simulation"]["num_subframes_per_frame"] = (
            self.subframes_per_frame
        )
        self.train_cfg["episode_length"] = steps
        self.config["hierarchical_parameters"]["timescale"]["subframes_per_frame"] = (
            self.subframes_per_frame
        )
        step_budget = self.train_cfg.get("total_steps")
        self._seed_everything(self.seed)
        self.env = (
            self.env
            if self.env is not None
            else HierISACEnv(self.config, bwp_mode=not self.single_bwp_mode)
        )
        self.env.max_steps = steps
        local, global_state = self.env.reset(seed=self.seed)
        local_dim, global_dim = self.contract.observation_dimensions(
            local, global_state
        )
        _, sac_dim = self.contract.sac_state(self.env.get_sac_state())
        self._create_missing_agent_components(sac_dim, local_dim)
        self.contract.sac_agent_structure(self.sac_agent, sac_dim)
        self.contract.dqn_agent_structure(self.dqn_agent, local_dim)
        manager = CheckpointManager(
            self.config,
            sac_agent=self.sac_agent,
            dqn_agent=self.dqn_agent,
            mode="hierarchical",
        )
        start, global_step = (
            (0, 0) if self.resume_from is None else manager.load(self.resume_from)
        )
        self.contract.finite_hier_parameters(self.sac_agent, self.dqn_agent)
        if step_budget is not None and global_step > step_budget:
            raise ValueError("Checkpoint exceeds requested physical step budget")
        if step_budget is None and start > episodes:
            raise ValueError("Checkpoint episode exceeds requested total")
        artifacts = ResultArtifacts(self.output_dir, run_name=self.model_name)
        metrics_path = artifacts.prepare_metrics(resume=self.resume_from is not None)
        mode = manager.prepare_metrics(metrics_path, start)
        history = []
        with metrics_path.open(mode, encoding="utf-8") as stream:
            for episode in range(
                start, step_budget if step_budget is not None else episodes
            ):
                if step_budget is not None and global_step >= step_budget:
                    break
                episode_limit = (
                    steps
                    if step_budget is None
                    else min(steps, step_budget - global_step)
                )
                scenario_seed = episode_scenario_seed(self.config, self.seed, episode)
                self.env.reset(seed=scenario_seed)
                pending = None
                sums, counts = {}, {}
                total_return, count = 0.0, 0
                sac_rewards, dqn_losses, sac_actor_losses, sac_critic_losses = (
                    [],
                    [],
                    [],
                    [],
                )
                terminated, truncated = False, False
                final_info = {}
                while count < episode_limit and not (terminated or truncated):
                    ss = self.env.get_sac_state()
                    raw = self.contract.sac_action(
                        self.sac_agent.select_action(ss), self.sac_agent.action_dim
                    )
                    frame_steps = min(self.subframes_per_frame, episode_limit - count)
                    self.env.begin_frame(
                        sac_ratios(raw, self.env.BWP_slice.shape, self.single_bwp_mode),
                        num_subframes=frame_steps,
                    )
                    # The last transition of a frame receives the next frame's
                    # actual quota before entering replay, never unlimited caps.
                    if pending is not None:
                        self._store_dqn(
                            pending,
                            self.env.get_dqn_state(),
                            self.env.budget,
                            global_step,
                            dqn_losses,
                        )
                        pending = None
                    for _ in range(frame_steps):
                        state = self.env.get_dqn_state()
                        budget = self.env.budget.copy()
                        action = self.contract.dqn_actions(
                            self.dqn_agent.select_action(
                                state, budget=budget, bwp_mode=not self.single_bwp_mode
                            )
                        )
                        output = self.contract.hier_step_result(
                            self.env.step(action), local_dim, global_dim
                        )
                        _, _, reward, terminated, env_truncated, final_info = output
                        self._accumulate_float_info(final_info, sums, counts)
                        total_return += reward
                        global_step += 1
                        count += 1
                        truncated = bool(env_truncated or count >= episode_limit)
                        pending = (state, action, reward, terminated, budget)
                        if not final_info["frame_complete"]:
                            self._store_dqn(
                                pending,
                                self.env.get_dqn_state(),
                                self.env.budget,
                                global_step,
                                dqn_losses,
                            )
                            pending = None
                        if self.epsilon_decay_unit == "subframe":
                            self.dqn_agent.decay_epsilon()
                        if terminated or truncated:
                            break
                    summary = self.env.last_frame_summary
                    if summary is None:
                        raise RuntimeError("Frame finished without a summary")
                    ss_next = self.env.get_sac_state()
                    sr = float(summary["reward"])
                    sac_rewards.append(sr)
                    self.sac_agent.replay_buffer.push(ss, raw, sr, ss_next, terminated)
                    for _ in range(
                        self.config["hierarchical_parameters"]["sac"].get(
                            "updates_per_frame", 1
                        )
                    ):
                        sm = self.sac_agent.update()
                        if sm["updated"]:
                            sac_critic_losses.append(sm["sac_critic_loss"])
                        if sm["actor_updated"]:
                            sac_actor_losses.append(sm["sac_actor_loss"])
                if pending is not None:
                    self._bootstrap_dqn(
                        pending, self.env.get_sac_state(), global_step, dqn_losses
                    )
                if self.epsilon_decay_unit == "episode":
                    self.dqn_agent.decay_epsilon()
                self.contract.finite_hier_parameters(self.sac_agent, self.dqn_agent)
                duration = count * self.config["system_parameters"]["slot_duration_ms"] / 1000.0
                record = {
                    **self._serializable_info(final_info),
                    **self._mean_float_info(sums, counts),
                    "episode": episode + 1,
                    "scenario_seed": scenario_seed,
                    "global_step": global_step,
                    "episode_return": float(total_return),
                    "episode_length": count,
                    "time_seconds": duration,
                    "episode_duration_seconds": duration,
                    "mean_sac_reward": float(np.mean(sac_rewards)),
                    "mean_dqn_reward": float(total_return / count),
                    "dqn_loss": float(np.mean(dqn_losses)) if dqn_losses else None,
                    "sac_actor_loss": float(np.mean(sac_actor_losses))
                    if sac_actor_losses
                    else None,
                    "sac_critic_loss": float(np.mean(sac_critic_losses))
                    if sac_critic_losses
                    else None,
                    "sac_alpha": self.sac_agent.alpha,
                    "dqn_epsilon": self.dqn_agent.epsilon,
                    "episode_dqn_updates": len(dqn_losses),
                    "episode_sac_critic_updates": len(sac_critic_losses),
                    "episode_sac_actor_updates": len(sac_actor_losses),
                    "sac_updates": self.sac_agent.update_step,
                    "dqn_updates": self.dqn_agent.update_step,
                }
                history.append(record)
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                if (episode + 1) % self.train_cfg["log_interval"] == 0:
                    print(
                        f"episode={episode + 1} steps={global_step} return={total_return:.6f} "
                        f"updates(dqn/sac_critic/sac_actor)="
                        f"{len(dqn_losses)}/{len(sac_critic_losses)}/{len(sac_actor_losses)}"
                    )
                interval = self.train_cfg["save_interval"]
                if (interval > 0 and (episode + 1) % interval == 0) or (
                    global_step >= step_budget
                    if step_budget is not None
                    else episode + 1 == episodes
                ):
                    manager.save(
                        self.output_dir
                        / "checkpoints"
                        / self.model_name
                        / f"episode_{episode + 1:06d}.pt",
                        episode,
                        global_step,
                    )
        artifacts.render(metrics_path)
        return history

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
            packet_count = key in ("urllc_completed_packets", "urllc_on_time_packets")
            if not (packet_count or isinstance(value, (float, np.floating))):
                continue
            if not np.isfinite(value):
                continue
            name = str(key)
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1

    @staticmethod
    def _mean_float_info(
        sums: Mapping[str, float], counts: Mapping[str, int]
    ) -> dict[str, float | int | None]:
        """Compute episode means for accumulated environment metrics."""
        result = {
            name: total / counts[name]
            for name, total in sums.items()
            if counts.get(name, 0) > 0
        }
        if "urllc_completed_packets" in sums:
            packets = sums["urllc_completed_packets"]
            result["mean_urllc_latency_ms"] = (
                sums.get("urllc_completed_latency_sum_ms", 0.0) / packets
                if packets
                else None
            )
            result["urllc_completed_packets"] = int(packets)
            result["urllc_on_time_packets"] = int(
                sums.get("urllc_on_time_packets", 0.0)
            )
            result["urllc_completed_latency_sum_ms"] = sums.get(
                "urllc_completed_latency_sum_ms", 0.0
            )
        return result
