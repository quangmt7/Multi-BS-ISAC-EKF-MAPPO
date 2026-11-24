"""Tracking references driven by the same multi-BWP Env transition kernel."""

from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo.utils.project_config import ProjectConfig
from ppo.envs.PPOenv import PPOISACEnv
from ppo.agents.PPOagent import MAPPOAgent
from ppo.utils.checkpoint_manager import CheckpointManager
from sensing.ekf import EKF
from sensing.pcrb import PCRB


class TrackingReference:
    """Conditional covariance reference using each actual aggregate observation once.

    The physical model combines BWP SINRs before generating one observation per
    BS/target. Repeating that observation as independent BWP measurements would
    double-count its Fisher information. The group explicitly records all BWPs.
    """

    def __init__(self, env):
        cfg = env.config["ekf_parameters"]
        args = (
            cfg["state_dim"],
            cfg["measurement_dim"],
            env.num_targets,
            cfg["dt"],
            env.scenario.get_process_noise_cov(),
            cfg["initial_covariance"],
        )
        self.group = "aggregate_bwps_" + "_".join(map(str, env.bwp))
        self.pcrb = PCRB(*args, False, bwp_ids=[self.group])
        self.local = [EKF(*args) for _ in range(env.num_bs)]
        initial = np.asarray(env.ekf_common.state_list).copy()
        self.pcrb.update_state(initial)
        for ekf in self.local:
            ekf.update_state(initial)

    def update(self, env):
        for m in range(env.num_targets):
            self.pcrb.predict(m)
            for ekf in self.local:
                ekf.predict(m)
        self.pcrb.set_linearization_states(env.target_states)
        for m, observations in enumerate(env.last_measurements):
            self.pcrb.update(
                m, [(self.group, env.bs_positions[b], z, r) for b, z, r in observations]
            )
            for b, ekf in enumerate(self.local):
                measurements = [
                    (env.bs_positions[b], z, r)
                    for origin, z, r in observations
                    if origin == b
                ]
                if measurements:
                    ekf.update(m, measurements)
        joint_p, joint_v = env.ekf_common.get_uncer()
        pcrb_p, pcrb_v = self.pcrb.get_uncer()
        local_p = np.asarray([ekf.get_uncer()[0] for ekf in self.local])
        local_v = np.asarray([ekf.get_uncer()[1] for ekf in self.local])
        return {
            "pos_joint": float(np.mean(joint_p)),
            "vel_joint": float(np.mean(joint_v)),
            "pos_pcrb": float(np.mean(pcrb_p)),
            "vel_pcrb": float(np.mean(pcrb_v)),
            "pos_best_bs": float(np.min(local_p, axis=0).mean()),
            "vel_best_bs": float(np.min(local_v, axis=0).mean()),
            "pos_mean_bs": float(local_p.mean()),
            "vel_mean_bs": float(local_v.mean()),
        }


class TrackingBenchmarkRunner:
    def __init__(self, config_dir=None, checkpoint_path=None, device="cpu"):
        root = Path(__file__).resolve().parents[1]
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("format_version") != 2 or checkpoint.get("mode") != "mappo":
                raise ValueError("Tracking requires a MAPPO checkpoint with the current observation schema")
        if checkpoint is not None and config_dir is None:
            project = ProjectConfig(checkpoint["config"], mode="mappo")
            project.validate()
            self.config = project.data
        else:
            self.config = ProjectConfig.from_directory(root / "config" if config_dir is None else config_dir).data
        self.config["mappo_parameters"]["training"]["device"] = str(device)
        self.env = PPOISACEnv(self.config)
        obs, state = self.env.reset(seed=42)
        self.max_steps = self.env.max_steps
        self.agent = None
        if checkpoint is not None:
            single = bool(checkpoint["single_bwp"])
            self.env.bwp_mode = not single
            self.agent = MAPPOAgent(
                self.config, obs.shape[1], state.size, single_bwp=single
            )
            CheckpointManager(self.config, ppo_agent=self.agent, mode="mappo").load(
                checkpoint_path
            )

    def _heuristic_allocation(self):
        env = self.env
        if not env.bwp_mode:
            entities = np.r_[np.arange(env.num_targets), env.num_targets + env.ue_indices[:, 0]]
            actions = np.zeros((env.num_bs, env.total_prb), dtype=np.int64)
            for b in range(env.num_bs):
                offset = 0
                for w, count in enumerate(env.nPrb):
                    destinations = np.flatnonzero((entities + b) % len(env.nPrb) == w) + 1
                    if destinations.size:
                        actions[b, offset:offset + count] = np.resize(destinations, count)
                    offset += count
            return actions
        # Balanced cyclic PRBs; all destinations use the current target-first map.
        destinations = 1 + env.num_targets + env.num_ue
        return np.asarray(
            [
                1 + (np.arange(env.total_prb) + b) % (destinations - 1)
                for b in range(env.num_bs)
            ],
            dtype=np.int64,
        )

    def run_benchmark(self, num_episodes=5, seed=42, use_heuristic=False):
        if (
            isinstance(num_episodes, bool)
            or not isinstance(num_episodes, int)
            or num_episodes <= 0
        ):
            raise ValueError("num_episodes must be positive")
        episodes = []
        failures = []
        for ep in range(num_episodes):
            obs, state = self.env.reset(seed=seed + ep)
            reference = TrackingReference(self.env)
            rows = []
            for _ in range(self.max_steps):
                if self.agent is not None and not use_heuristic:
                    actions = self.agent.select_action(obs, state, deterministic=True)[
                        0
                    ]
                else:
                    actions = self._heuristic_allocation()
                obs, state, _, terminated, truncated, info = self.env.step(actions)
                rows.append(
                    {
                        **reference.update(self.env),
                        "comm_rate_mean_mbps": info["comm_rate_mean_mbps"],
                        "sensing_success_prob": info["sensing_success_prob_mean"],
                    }
                )
                if terminated or truncated:
                    break
            failures.append(bool(terminated))
            episodes.append(rows)
        keys = episodes[0][0]
        length = max(map(len, episodes))
        counts = np.zeros(length, dtype=int)
        totals = {key: np.zeros(length) for key in keys}
        for episode in episodes:
            counts[:len(episode)] += 1
            for key in keys:
                totals[key][:len(episode)] += [row[key] for row in episode]
        results = {key: (values / counts).tolist() for key, values in totals.items()}
        results.update(
            slots=(np.arange(length) + 1).tolist(),
            contributing_episodes=counts.tolist(),
            terminal_failures=failures,
            measurement_model="one aggregate observation per detected BS/target",
            reference="conditional linearized covariance recursion; not an optimal policy",
            seeds=list(range(seed, seed + num_episodes)),
        )
        return results

    def save_and_plot(self, results, output_dir=None):
        import matplotlib.pyplot as plt

        output = (
            Path(__file__).resolve().parents[1] / "results" / "tracking"
            if output_dir is None
            else Path(output_dir)
        )
        if any((output / name).exists() for name in (
            "benchmark_tracking_results.json", "tracking_pos_uncertainty.png", "tracking_vel_uncertainty.png"
        )):
            raise FileExistsError(f"Tracking results already exist in {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "benchmark_tracking_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        for kind, unit in (("pos", "m^2"), ("vel", "m^2/s^2")):
            fig, ax = plt.subplots(figsize=(10, 5))
            for suffix, label in (
                ("mean_bs", "Local EKF mean"),
                ("best_bs", "Local EKF best"),
                ("joint", "Joint EKF"),
                ("pcrb", "Conditional covariance reference"),
            ):
                ax.plot(results["slots"], results[f"{kind}_{suffix}"], label=label)
            ax.set(
                xlabel="Physical step (1 ms)",
                ylabel=f"Covariance trace ({unit})",
                yscale="log",
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"tracking_{kind}_uncertainty.png", dpi=160)
            plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Tracking references on the multi-BWP physical kernel"
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "tracking")
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args(argv)
    runner = TrackingBenchmarkRunner(args.config_dir, args.checkpoint)
    results = runner.run_benchmark(args.episodes, args.seed, args.heuristic)
    runner.save_and_plot(results, args.out_dir)


if __name__ == "__main__":
    main()
