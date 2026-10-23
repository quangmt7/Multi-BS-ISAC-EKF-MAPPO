"""
Tracking Benchmark Module for ISAC System:
Comparing 3 Curves over Time Slots:
1. Joint Cooperative EKF (Centralized Multi-BS Fusion)
2. Each-BS Local EKF (Decentralized Tracking per Base Station)
3. PCRB (Posterior Cramer-Rao Bound - Theoretical Lower Limit)
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ppo.utils.project_config import ProjectConfig
from scenario.scenario import Scenario
from channel.channel import WirelessChannel
from sensing.detection import Detection
from sensing.bs_measurement import BSMeasurement
from sensing.ekf import EKF
from sensing.pcrb import PCRB
from ppo.AgentComponents import Actor
from communication.communication import CommunicationManager


class TrackingBenchmarkRunner:
    def __init__(self, config_dir="d:/Lab/ISAC/isac/config", checkpoint_path=None, device="cpu"):
        self.device = device
        
        # Load project configuration
        self.config = ProjectConfig.from_directory(config_dir).data
        sys_config = self.config["system_parameters"]
        self.ekf_cfg = self.config["ekf_parameters"]
        sim_config = sys_config["simulation"]
        scenario_config = sys_config["scenario"]

        self.num_bs = sys_config["num_bs"]
        self.num_ues = sys_config["num_ue"]
        self.num_targets = sys_config["num_target"]
        self.n_prb = sys_config["num_prb_bs"]
        self.p_max = float(sys_config["max_tx_power_watt"])
        self.max_steps = sim_config["num_slots_per_episode"]
        
        lx = float(scenario_config["area_size_x_m"])
        ly = float(scenario_config["area_size_y_m"])
        lz = float(scenario_config["area_size_z_m"])
        self.d_max = float(np.sqrt(lx**2 + ly**2 + lz**2))

        # Normalization scales for observation construction (Direct key indexing - fail fast)
        norm_cfg = self.config["mappo_parameters"]["normalization"]
        self.a_max = float(norm_cfg["a_max"])
        self.u_pos_max = float(self.ekf_cfg["uncertainty_limits"]["u_pos_max"])
        self.u_vel_max = float(self.ekf_cfg["uncertainty_limits"]["u_vel_max"])
        self.se_comm_scale = float(np.log2(1.0 + sys_config["sinr"]["gamma_c_min_linear"]))
        self.se_sens_scale = float(np.log2(1.0 + sys_config["sinr"]["gamma_s_min_linear"]))

        # Instantiate physical modules
        self.scenario = Scenario(self.config)
        self.channel = WirelessChannel(self.config)
        p_fa = float(sys_config["requirements"]["max_prob_false_alarm"])
        self.detection_list = [Detection(p_fa) for _ in range(self.num_bs)]
        self.meas_list = [BSMeasurement(b, self.ekf_cfg) for b in range(self.num_bs)]
        self.comm_managers = [CommunicationManager(self.config, bs_id=b) for b in range(self.num_bs)]

        # Instantiate 3 Tracking Models:
        # 1. Joint Cooperative EKF (Centralized Multi-BS fusion)
        self.ekf_joint = EKF(
            self.ekf_cfg["state_dim"], self.ekf_cfg["measurement_dim"],
            self.num_targets, self.ekf_cfg["dt"],
            self.ekf_cfg["process_noise"], self.ekf_cfg["initial_covariance"], False
        )
        
        # 2. Each-BS Local EKF (One independent instance per BS)
        self.ekf_separate = [
            EKF(
                self.ekf_cfg["state_dim"], self.ekf_cfg["measurement_dim"],
                self.num_targets, self.ekf_cfg["dt"],
                self.ekf_cfg["process_noise"], self.ekf_cfg["initial_covariance"], True
            )
            for _ in range(self.num_bs)
        ]

        # 3. PCRB (Theoretical Cramér-Rao lower bound)
        self.pcrb = PCRB(
            self.ekf_cfg["state_dim"], self.ekf_cfg["measurement_dim"],
            self.num_targets, self.ekf_cfg["dt"],
            self.ekf_cfg["process_noise"], self.ekf_cfg["initial_covariance"], False
        )

        # Load Trained Actor if checkpoint provided
        self.actor = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] Loading trained Actor checkpoint: {checkpoint_path}")
            local_obs_dim = 8 * self.num_targets + self.num_ues + 8
            actor_conf = self.config["mappo_parameters"]["network"]["actor"]
            self.actor = Actor(
                local_obs_dim=local_obs_dim,
                n_prb=self.n_prb,
                num_targets=self.num_targets,
                num_ues=self.num_ues,
                net_conf=actor_conf
            ).to(self.device)

            chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "actor" not in chk:
                raise KeyError(f"Key 'actor' not found in checkpoint '{checkpoint_path}'! Checkpoint keys: {list(chk.keys())}")
            self.actor.load_state_dict(chk["actor"])
            print("[OK] Actor policy weights successfully loaded! Running in Policy-Guided Mode.")
            self.actor.eval()
        else:
            print("[INFO] Running in Baseline Mode (Multi-Target Balanced Heuristic).")

    def reset_sim(self, seed=42):
        np.random.seed(seed)
        scen_data = self.scenario.reset(seed=seed)
        self.bs_positions = scen_data["bs_positions"]
        self.ue_positions = scen_data["ue_positions"]
        self.target_states = scen_data["target_states"].copy()

        # Update initial states in all 3 models
        self.ekf_joint.update_state(self.target_states)
        self.pcrb.update_state(self.target_states)
        for b in range(self.num_bs):
            self.meas_list[b].update_bs_pos(self.bs_positions[b])
            self.ekf_separate[b].update_state(self.target_states)

        self.current_step = 0
        self.aoi = np.zeros(self.num_targets, dtype=np.float32)
        self.p_res = np.full(self.num_bs, self.p_max, dtype=np.float32)
        self.n_res = np.full(self.num_bs, self.n_prb, dtype=np.float32)
        self.last_h_comm = np.zeros((self.num_bs, self.num_ues), dtype=np.float32)
        self.last_h_sens = np.zeros((self.num_bs, self.num_targets), dtype=np.float32)
        self.pred_target_states = self.target_states.copy()
        if hasattr(self.channel, "rng"):
            self.channel.rng = np.random.default_rng(seed)

    def _get_local_observations(self):
        """Build local observation vector for each BS b."""
        u_pos, u_vel = self.ekf_joint.get_uncer()
        u_pos_norm = np.clip(u_pos / self.u_pos_max, 0.0, 1.0).astype(np.float32)
        u_vel_norm = np.clip(u_vel / self.u_vel_max, 0.0, 1.0).astype(np.float32)
        aoi_norm = np.clip(self.aoi / self.a_max, 0.0, 1.0).astype(np.float32)

        pred_target_pos = self.pred_target_states[:, :3]
        pred_target_pos_norm = np.clip(pred_target_pos / self.d_max, -1.0, 1.0).flatten().astype(np.float32)

        t_frac = float(self.current_step) / float(max(1, self.max_steps))
        time_features = np.array([
            t_frac,
            np.sin(2.0 * np.pi * t_frac),
            np.cos(2.0 * np.pi * t_frac)
        ], dtype=np.float32)

        obs_list = []
        for b in range(self.num_bs):
            d_pred_b = np.zeros(self.num_targets, dtype=np.float32)
            for m in range(self.num_targets):
                dist = float(np.linalg.norm(pred_target_pos[m] - self.bs_positions[b]))
                d_pred_b[m] = np.clip(dist / self.d_max, 0.0, 1.0)

            h_comm_b = np.clip(self.last_h_comm[b] / self.se_comm_scale, 0.0, 1.0).astype(np.float32)
            h_sens_b = np.clip(self.last_h_sens[b] / self.se_sens_scale, 0.0, 1.0).astype(np.float32)
            bs_pos_norm = np.clip(self.bs_positions[b] / self.d_max, -1.0, 1.0).astype(np.float32)
            p_res_b = np.array([np.clip(self.p_res[b] / self.p_max, 0.0, 1.0)], dtype=np.float32)
            n_res_b = np.array([np.clip(self.n_res[b] / self.n_prb, 0.0, 1.0)], dtype=np.float32)

            obs_b = np.concatenate([
                u_pos_norm,
                u_vel_norm,
                aoi_norm,
                pred_target_pos_norm,
                d_pred_b,
                h_comm_b,
                h_sens_b,
                bs_pos_norm,
                p_res_b,
                n_res_b,
                time_features
            ]).astype(np.float32)
            obs_list.append(obs_b)

        return np.stack(obs_list, axis=0)

    def action_mapping(self, actions_prb, actions_pwr):
        """Map discrete PRB choices and continuous power to X_s, X_c, P_s, P_c."""
        if hasattr(actions_prb, "detach"):
            actions_prb = actions_prb.detach().cpu().numpy()
        if hasattr(actions_pwr, "detach"):
            actions_pwr = actions_pwr.detach().cpu().numpy()

        actions_prb = np.asarray(actions_prb, dtype=np.int32)
        actions_pwr = np.asarray(actions_pwr, dtype=np.float32)

        X_s = np.zeros((self.num_bs, self.num_targets, self.n_prb), dtype=np.float32)
        X_c = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)
        P_s = np.zeros((self.num_bs, self.num_targets, self.n_prb), dtype=np.float32)
        P_c = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)

        for b in range(self.num_bs):
            act_prb_b = actions_prb[b]
            raw_pwr_b = np.clip(actions_pwr[b], 0.0, 1.0)
            active_mask = (act_prb_b > 0)
            self.n_res[b] = float(np.sum(~active_mask))

            pwr_allocated = np.zeros(self.n_prb, dtype=np.float32)
            if np.any(active_mask):
                p_prb_nominal = self.p_max / self.n_prb
                # Each active PRB receives power proportional to raw_pwr_b (up to 2 * nominal with power boosting)
                pwr_allocated[active_mask] = raw_pwr_b[active_mask] * (2.0 * p_prb_nominal)
                
                # Enforce total BS power budget P_max
                total_requested_pwr = float(np.sum(pwr_allocated))
                if total_requested_pwr > self.p_max:
                    pwr_allocated[active_mask] *= (self.p_max / total_requested_pwr)

            for r in range(self.n_prb):
                dest = int(act_prb_b[r])
                val_power = pwr_allocated[r]
                if dest == 0:
                    continue
                elif 1 <= dest <= self.num_targets:
                    m = dest - 1
                    X_s[b, m, r] = 1.0
                    P_s[b, m, r] = val_power
                elif self.num_targets < dest <= self.num_targets + self.num_ues:
                    k = dest - self.num_targets - 1
                    X_c[b, k, r] = 1.0
                    P_c[b, k, r] = val_power

            used_power = float(np.sum(P_s[b]) + np.sum(P_c[b]))
            self.p_res[b] = max(0.0, self.p_max - used_power)

        return X_s, X_c, P_s, P_c

    def _heuristic_allocation(self, sens_prb_ratio=0.4):
        """
        Balanced multi-target, orthogonal OFDMA baseline allocation.
        - Equitably divides sensing PRBs across ALL targets (no target left behind).
        - Equitably divides remaining communication PRBs across ALL UEs orthogonally.
        """
        X_s = np.zeros((self.num_bs, self.num_targets, self.n_prb), dtype=np.float32)
        X_c = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)
        P_s = np.zeros((self.num_bs, self.num_targets, self.n_prb), dtype=np.float32)
        P_c = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)

        total_sens_prb = int(self.n_prb * sens_prb_ratio)
        prbs_per_target = total_sens_prb // self.num_targets
        p_per_prb = self.p_max / self.n_prb

        # 1. Sensing allocation across all targets
        for b in range(self.num_bs):
            for m in range(self.num_targets):
                start_r = m * prbs_per_target
                end_r = (m + 1) * prbs_per_target
                X_s[b, m, start_r:end_r] = 1.0
                P_s[b, m, start_r:end_r] = p_per_prb

        # 2. Orthogonal OFDMA communication allocation across all UEs
        comm_prb_start = self.num_targets * prbs_per_target
        remaining_comm_prbs = self.n_prb - comm_prb_start
        prbs_per_ue = remaining_comm_prbs // self.num_ues

        for b in range(self.num_bs):
            for k in range(self.num_ues):
                start_k = comm_prb_start + k * prbs_per_ue
                end_k = comm_prb_start + (k + 1) * prbs_per_ue if k < self.num_ues - 1 else self.n_prb
                X_c[b, k, start_k:end_k] = 1.0
                P_c[b, k, start_k:end_k] = p_per_prb

            self.p_res[b] = 0.0
            self.n_res[b] = 0.0

        return X_s, X_c, P_s, P_c

    def run_benchmark(self, num_episodes=5, seed=42, use_heuristic=False):
        """Run benchmark evaluation and collect tracking trajectories."""
        mode_str = "Policy-Guided (MAPPO Actor)" if (self.actor is not None and not use_heuristic) else "Balanced Multi-Target Heuristic"
        print(f"\n>>> Running Tracking Benchmark: Mode [{mode_str}] over {num_episodes} episodes ({self.max_steps} slots each)...")
        
        pos_joint = np.zeros((num_episodes, self.max_steps))
        vel_joint = np.zeros((num_episodes, self.max_steps))
        
        pos_best_bs = np.zeros((num_episodes, self.max_steps))
        vel_best_bs = np.zeros((num_episodes, self.max_steps))

        pos_mean_bs = np.zeros((num_episodes, self.max_steps))
        vel_mean_bs = np.zeros((num_episodes, self.max_steps))

        pos_pcrb = np.zeros((num_episodes, self.max_steps))
        vel_pcrb = np.zeros((num_episodes, self.max_steps))

        comm_rate_ep = np.zeros((num_episodes, self.max_steps))
        p_suc_ep = np.zeros((num_episodes, self.max_steps))

        for ep in range(num_episodes):
            self.reset_sim(seed=seed + ep * 100)
            
            for t in range(self.max_steps):
                self.current_step = t
                # 1. Target dynamic motion step (Synchronized with EKF predict at every slot)
                self.scenario.step_targets()
                self.target_states = self.scenario.get_target_states()
                target_pos = self.target_states[:, :3]

                # Update PCRB linearization ground-truth
                self.pcrb.set_linearization_states(self.target_states)

                # 2. Kalman Predict Step for all models (Prior state for beam steering)
                for m in range(self.num_targets):
                    self.ekf_joint.predict(m)
                    self.pcrb.predict(m)
                    for b in range(self.num_bs):
                        self.ekf_separate[b].predict(m)

                self.pred_target_states = np.asarray(self.ekf_joint.state_list, dtype=np.float32)
                pred_target_pos = self.pred_target_states[:, :3]

                # 3. Decide Action: Trained Policy or Multi-Target Heuristic
                if self.actor is not None and not use_heuristic:
                    obs = self._get_local_observations()
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                        feat = self.actor.backbone(obs_t)
                        prb_logits = self.actor.prb_head(feat).reshape(-1, self.actor.n_prb, self.actor.num_choices)
                        # Deterministic policy action
                        actions_prb = torch.argmax(prb_logits, dim=-1)
                        raw_mean = self.actor.power_mean(feat)
                        actions_pwr = torch.sigmoid(raw_mean)
                    X_s, X_c, P_s, P_c = self.action_mapping(actions_prb, actions_pwr)
                else:
                    X_s, X_c, P_s, P_c = self._heuristic_allocation()

                # 4. Realistic Channel computation (Beam steering directed at predicted positions)
                channel_out = self.channel.step(
                    P_c=P_c,
                    P_s=P_s,
                    X_c=X_c,
                    X_s=X_s,
                    bs_pos=self.bs_positions,
                    ue_pos=self.ue_positions,
                    target_pos=target_pos,
                    pred_target_pos=pred_target_pos
                )
                effective_sinr = channel_out["effective_sensing_sinr"]
                comm_gain = channel_out["comm_gain"]

                # Update Communication Throughput and channel features H_avg
                user_rates = np.zeros(self.num_ues, dtype=np.float32)
                comm_power_by_bs = np.sum(P_c, axis=1)
                for b in range(self.num_bs):
                    x_b = X_c[b]
                    p_b = P_c[b]
                    gain_b = comm_gain[b]
                    other_bs_interf = self.channel.ibi_eta * np.sum(
                        comm_power_by_bs[np.arange(self.num_bs) != b, np.newaxis, :]
                        * comm_gain[np.arange(self.num_bs) != b], axis=0
                    )
                    sinr_b, se_b = self.comm_managers[b].compute_sinr(
                        x_b, p_b, gain_b, inter_bs_interf=other_bs_interf
                    )
                    rate_b = self.comm_managers[b].compute_user_rates(x_b, se_b)
                    user_rates += rate_b
                    self.last_h_comm[b] = np.mean(se_b, axis=1).astype(np.float32)

                self.last_h_sens = np.mean(
                    np.log2(1.0 + np.maximum(channel_out["sensing_sinr_prb"], 0.0)), axis=2
                ).astype(np.float32)

                # 5. Radar detection step
                detect_mask = np.zeros((self.num_bs, self.num_targets))
                miss_prob = np.ones(self.num_targets)
                for m in range(self.num_targets):
                    for b in range(self.num_bs):
                        if effective_sinr[b, m] > 0:
                            detected, p_d = self.detection_list[b].detect(effective_sinr[b, m])
                            miss_prob[m] *= (1.0 - p_d)
                            if detected:
                                detect_mask[b, m] = 1

                # Update Age of Information (AoI)
                for m in range(self.num_targets):
                    if np.any(detect_mask[:, m] == 1):
                        self.aoi[m] = 0.0
                    else:
                        self.aoi[m] += 1.0

                # 6. Generate radar measurements
                measurements_by_target = [[] for _ in range(self.num_targets)]
                for m in range(self.num_targets):
                    for b in range(self.num_bs):
                        if detect_mask[b, m] == 1:
                            meas, cov = self.meas_list[b].gen_meas(self.target_states[m], effective_sinr[b, m])
                            measurements_by_target[m].append((self.bs_positions[b], meas, cov))

                # 7. Update 3 tracking models with measurements:
                for m in range(self.num_targets):
                    # (a) PCRB update (Centralized cooperative fusion)
                    if measurements_by_target[m]:
                        self.pcrb.update(m, measurements_by_target[m])

                    # (b) Joint Cooperative EKF update (Centralized fusion)
                    if measurements_by_target[m]:
                        self.ekf_joint.update(m, measurements_by_target[m])

                    # (c) Each-BS Local EKF LOCAL update ONLY
                    for b in range(self.num_bs):
                        if detect_mask[b, m] == 1:
                            local_meas = [(pos, meas, cov) for pos, meas, cov in measurements_by_target[m]
                                          if np.array_equal(pos, self.bs_positions[b])]
                            if local_meas:
                                self.ekf_separate[b].update(m, local_meas)

                # Record Uncertainty at time t
                # (a) Joint EKF
                uj_p, uj_v = self.ekf_joint.get_uncer()
                pos_joint[ep, t] = np.mean(uj_p)
                vel_joint[ep, t] = np.mean(uj_v)

                # (b) PCRB (Theoretical lower limit)
                up_p, up_v = self.pcrb.get_uncer()
                pos_pcrb[ep, t] = np.mean(up_p)
                vel_pcrb[ep, t] = np.mean(up_v)

                # (c) Each-BS EKF: Best-BS (min) & Mean-BS
                bs_pos_uncer = [self.ekf_separate[b].get_uncer()[0] for b in range(self.num_bs)]
                bs_vel_uncer = [self.ekf_separate[b].get_uncer()[1] for b in range(self.num_bs)]
                pos_best_bs[ep, t] = np.mean(np.min(bs_pos_uncer, axis=0))
                vel_best_bs[ep, t] = np.mean(np.min(bs_vel_uncer, axis=0))
                pos_mean_bs[ep, t] = np.mean(np.mean(bs_pos_uncer, axis=0))
                vel_mean_bs[ep, t] = np.mean(np.mean(bs_vel_uncer, axis=0))

                # Record Communication Throughput & Sensing Success
                user_rates_mbps = user_rates / 1e6
                comm_rate_ep[ep, t] = np.mean(user_rates_mbps)
                p_suc_ep[ep, t] = np.mean(1.0 - miss_prob)

        # Average across episodes
        benchmark_results = {
            "slots": (np.arange(self.max_steps) + 1).tolist(),
            "pos_pcrb": np.mean(pos_pcrb, axis=0).tolist(),
            "pos_joint": np.mean(pos_joint, axis=0).tolist(),
            "pos_best_bs": np.mean(pos_best_bs, axis=0).tolist(),
            "pos_mean_bs": np.mean(pos_mean_bs, axis=0).tolist(),
            "vel_pcrb": np.mean(vel_pcrb, axis=0).tolist(),
            "vel_joint": np.mean(vel_joint, axis=0).tolist(),
            "vel_best_bs": np.mean(vel_best_bs, axis=0).tolist(),
            "vel_mean_bs": np.mean(vel_mean_bs, axis=0).tolist(),
            "comm_rate_mean_mbps": np.mean(comm_rate_ep, axis=0).tolist(),
            "sensing_success_prob": np.mean(p_suc_ep, axis=0).tolist(),
        }

        return benchmark_results

    def save_and_plot(self, results, output_dir="d:/Lab/ISAC/isac/benchmark/results"):
        os.makedirs(output_dir, exist_ok=True)
        slots = results["slots"]

        # 1. Save Raw Data JSON
        json_path = os.path.join(output_dir, "benchmark_tracking_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Benchmark data successfully saved to: {json_path}")

        # 2. Plot Position Uncertainty Comparison (3 Curves)
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        ax.plot(slots, results["pos_mean_bs"], label="Each-BS EKF (Average BS)", color="#ff7f0e", linestyle="-.", linewidth=1.8, alpha=0.8)
        ax.plot(slots, results["pos_best_bs"], label="Each-BS EKF (Best Local BS)", color="#d62728", linestyle="--", linewidth=2.2)
        ax.plot(slots, results["pos_joint"], label="Joint Cooperative EKF (Ours)", color="#1f77b4", linewidth=2.5)
        ax.plot(slots, results["pos_pcrb"], label="PCRB (Theoretical Lower Bound)", color="black", linestyle="--", linewidth=2.2)

        ax.set_yscale("log")
        ax.set_xlabel("Time Slot (t)", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Position Uncertainty $\mathrm{Tr}(\mathbf{P}_{\mathrm{pos}})$ ($\mathrm{m}^2$) [Log Scale]", fontsize=12, fontweight="bold")
        ax.set_title("Target Position Tracking Uncertainty: PCRB vs. Joint EKF vs. Each-BS EKF (Log Scale)", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11, loc="upper right", frameon=True)
        fig_pos_path = os.path.join(output_dir, "tracking_pos_uncertainty_3curves.png")
        plt.savefig(fig_pos_path, bbox_inches="tight")
        plt.close()

        # 3. Plot Velocity Uncertainty Comparison (3 Curves)
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        ax.plot(slots, results["vel_mean_bs"], label="Each-BS EKF (Average BS)", color="#ff7f0e", linestyle="-.", linewidth=1.8, alpha=0.8)
        ax.plot(slots, results["vel_best_bs"], label="Each-BS EKF (Best Local BS)", color="#d62728", linestyle="--", linewidth=2.2)
        ax.plot(slots, results["vel_joint"], label="Joint Cooperative EKF (Ours)", color="#1f77b4", linewidth=2.5)
        ax.plot(slots, results["vel_pcrb"], label="PCRB (Theoretical Lower Bound)", color="black", linestyle="--", linewidth=2.2)

        ax.set_yscale("log")
        ax.set_xlabel("Time Slot (t)", fontsize=12, fontweight="bold")
        ax.set_ylabel(r"Velocity Uncertainty $\mathrm{Tr}(\mathbf{P}_{\mathrm{vel}})$ ($\mathrm{m}^2/\mathrm{s}^2$) [Log Scale]", fontsize=12, fontweight="bold")
        ax.set_title("Target Velocity Tracking Uncertainty: PCRB vs. Joint EKF vs. Each-BS EKF (Log Scale)", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11, loc="upper right", frameon=True)
        fig_vel_path = os.path.join(output_dir, "tracking_vel_uncertainty_3curves.png")
        plt.savefig(fig_vel_path, bbox_inches="tight")
        plt.close()

        # Print Numerical Summary
        avg_pcrb = np.mean(results["pos_pcrb"])
        avg_joint = np.mean(results["pos_joint"])
        avg_best_bs = np.mean(results["pos_best_bs"])
        gain_vs_each = (avg_best_bs - avg_joint) / avg_best_bs * 100
        gap_to_pcrb = (avg_joint - avg_pcrb) / avg_pcrb * 100

        print("\n================ BENCHMARK NUMERICAL SUMMARY ================")
        print(f"1. PCRB Bound Pos Uncertainty      : {avg_pcrb:.8f} m^2 (Theoretical Optimum)")
        print(f"2. Joint Cooperative EKF Pos Uncer  : {avg_joint:.8f} m^2")
        print(f"3. Each-BS Local EKF Pos Uncer     : {avg_best_bs:.8f} m^2")
        print(f"--> Joint EKF reduces uncertainty by: {gain_vs_each:.8f}% compared to Each-BS EKF!")
        print(f"--> Gap between Joint EKF and PCRB  : only {gap_to_pcrb:.8f}%!")
        print("=============================================================\n")
        print(f"Generated figures in {output_dir}:")
        print(f"- {fig_pos_path}")
        print(f"- {fig_vel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAC 3-Curve Tracking Benchmark Runner")
    parser.add_argument("--config-dir", type=str, default="d:/Lab/ISAC/isac/config")
    parser.add_argument("--checkpoint", type=str, default="d:/Lab/ISAC/isac/results/checkpoints/episode_002000.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default="d:/Lab/ISAC/isac/benchmark/results")
    parser.add_argument("--heuristic", action="store_true", help="Force baseline heuristic allocation instead of policy")
    args = parser.parse_args()

    runner = TrackingBenchmarkRunner(
        config_dir=args.config_dir,
        checkpoint_path=args.checkpoint if not args.heuristic else None
    )
    res = runner.run_benchmark(num_episodes=args.episodes, seed=args.seed, use_heuristic=args.heuristic)
    runner.save_and_plot(res, output_dir=args.out_dir)
