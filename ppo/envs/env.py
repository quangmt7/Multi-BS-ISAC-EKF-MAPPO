"""Shared physical transition kernel for the PPO and hierarchical environments."""
from abc import ABC, abstractmethod

import numpy as np

from scenario.scenario import Scenario
from channel.channel import WirelessChannel
from communication.communication import CommunicationManager
from sensing.detection import Detection
from sensing.bs_measurement import BSMeasurement
from sensing.ekf import EKF
from ppo.utils.project_config import ProjectConfig


class BaseISACEnv(ABC):
    """One step advances all BSs by slot_duration_ms, exactly once.

    Actions are integer destinations [B,W,Rmax] (padding must be idle), or
    [B,sum(nPrb)] in BWP order. 0=idle, 1..M=target, M+1..M+K=global UE.
    Scenario.ues defines the global UE order; padded UEs have no destination.
    pEach is configured in mW; internal power/energy use W/J.
    BWP_slice, budgets and SAC ratio columns store targets before slices.
    cost_max, when supplied, contains GLOBAL per-step bounds in the order
    energy, switching, guardband.
    """

    def __init__(self, config: dict, cost_max=None, bwp_mode=True):
        ProjectConfig(config).validate()
        self.config = config
        self.bwp_mode = bool(bwp_mode)
        sys_config = config["system_parameters"]
        req_config = sys_config["requirements"]
        ekf_config = config["ekf_parameters"]
        mappo_cfg = config["mappo_parameters"]
        self.num_bs = sys_config["num_bs"]
        self.num_embb = sys_config["num_embb"]
        self.num_urllc = sys_config["num_urllc"]
        self.num_slices = self.num_embb + self.num_urllc
        self.num_ue_1embb = sys_config["num_ue_1embb"]
        self.num_ue_1urllc = sys_config["num_ue_1urllc"]
        self.num_targets = sys_config["num_target"]
        self.num_ue = self.num_embb * self.num_ue_1embb + self.num_urllc * self.num_ue_1urllc
        self.bwp = list(sys_config["bwp"])
        self.nPrb = list(sys_config["nPrb"])
        self.max_prb = max(self.nPrb)
        self.total_prb = sum(self.nPrb)
        self.prb_mask = np.arange(self.max_prb)[None, :] < np.asarray(self.nPrb)[:, None]
        self.pEach = np.asarray(sys_config["pEach"], dtype=float) * 1e-3
        self.p_max = float(np.dot(self.nPrb, self.pEach))
        self.prbBW = float(sys_config["prbBW"])  # config kHz
        self.slot_duration = float(sys_config["slot_duration_ms"]) / 1000.
        self.max_steps = sys_config["simulation"]["num_slots_per_episode"]
        self.cost_switch_c = float(sys_config["cost_switch_c"])
        self.cost_switch_s = float(sys_config["cost_switch_s"])
        self.pa_efficiency = float(sys_config["pa_eff"])
        self.dt = float(sys_config["dt"])
        self.cost_gb = float(sys_config["cost_gb"])
        self.required_bits_urllc = float(sys_config["urllc_packet_size_bits"])
        self.r_min = float(req_config["min_ue_rate_mbps"])
        self.max_ue_delay = float(req_config["max_ue_delay"])  # ms, separate from geometry
        self.p_min_sensing = float(req_config["min_sensing_success_prob"])
        self.p_d_min = float(req_config["min_detection_prob"])
        self.p_fa = float(req_config["max_prob_false_alarm"])
        geometry = sys_config["scenario"]
        self.d_max = float(np.linalg.norm([geometry["area_size_x_m"],
                                          geometry["area_size_y_m"], geometry["area_size_z_m"]]))
        reward_cfg = mappo_cfg["reward_weights"]
        self.alpha = float(reward_cfg["alpha_aoi"])
        self.beta = float(reward_cfg["beta_uncertainty"])
        self.gamma = float(reward_cfg["gamma_power"])
        self.lambda_service = float(reward_cfg["lambda_service"])
        shaping = mappo_cfg["reward_shaping"]
        self.reward_k = np.array([shaping["k_aoi"], shaping["k_uncertainty"], shaping["k_service"]], dtype=float)
        if not np.isfinite([self.alpha, self.beta, self.gamma, self.lambda_service]).all() or min(self.alpha, self.beta, self.gamma, self.lambda_service) < 0:
            raise ValueError("reward weights must be finite and non-negative")
        if not np.isfinite(self.reward_k).all() or np.any(self.reward_k <= 0):
            raise ValueError("reward shaping coefficients must be positive")
        self.a_max = float(mappo_cfg["normalization"]["a_max"])
        self.u_pos_max = float(ekf_config["uncertainty_limits"]["u_pos_max"])
        self.u_vel_max = float(ekf_config["uncertainty_limits"]["u_vel_max"])
        if min(self.a_max, self.u_pos_max, self.u_vel_max, self.d_max,
               self.max_ue_delay, self.p_min_sensing) <= 0:
            raise ValueError("normalization scales and QoS thresholds must be positive")
        penalties = mappo_cfg["penalties"]
        self.c2 = float(penalties.get("c2_sensing_reliability", 0.))
        self.c3 = float(penalties.get("c3_pos_uncertainty", 0.))
        self.c4 = float(penalties.get("c4_vel_uncertainty", 0.))
        self.c5 = float(penalties.get("c5_sens_freq", 0.))
        self.aoi_thr = float(penalties.get("aoi_thr", 0.))
        self.se_comm_scale = float(np.log2(1 + sys_config["sinr"]["gamma_c_min_linear"]))
        self.se_sens_scale = float(np.log2(1 + sys_config["sinr"]["gamma_s_min_linear"]))

        self.ekf_config = ekf_config
        self.scenario = Scenario(config, cfg_ekf=ekf_config)
        self.channel = WirelessChannel(config)
        # CommunicationManager's API aggregates all BSs; call it once per step.
        self.comm_managers = [CommunicationManager(config, b) for b in range(self.num_bs)]
        self.detection_list = [Detection(self.p_fa) for _ in range(self.num_bs)]
        self.measurement_list = [BSMeasurement(b, ekf_config) for b in range(self.num_bs)]
        self.ekf_common = EKF(
            ekf_config["state_dim"], ekf_config["measurement_dim"], self.num_targets,
            ekf_config["dt"], self.scenario.get_process_noise_cov(), ekf_config["initial_covariance"])
        self.numBit = np.zeros((self.num_urllc, self.num_ue_1urllc), dtype=float)
        self.URLLC_Latency = np.zeros_like(self.numBit)
        self.eMBB_Thr = np.zeros((self.num_embb, self.num_ue_1embb), dtype=float)
        self.BWP_slice = np.zeros((self.num_bs, len(self.bwp),
                                   self.num_targets + self.num_slices), dtype=np.int8)
        self.last_BWP_slice = np.zeros_like(self.BWP_slice)
        # Loose upper bounds, all at the same (global, per-step) aggregation level.
        frequency_order = np.argsort(sys_config["fStart"], kind="stable")
        self.adjacent_bwp_pairs = list(zip(frequency_order[:-1], frequency_order[1:]))
        guard_pair_sum = sum(np.ceil(self.cost_gb * abs(self.bwp[w] - self.bwp[v]))
                             for w, v in self.adjacent_bwp_pairs)
        self.cost_max_local = np.array([
            self.p_max * self.dt / self.pa_efficiency,
            len(self.bwp) * (self.num_slices * self.cost_switch_c + self.num_targets * self.cost_switch_s),
            guard_pair_sum], dtype=float)
        self.cost_max = (self.cost_max_local * self.num_bs if cost_max is None
                         else np.asarray(cost_max, dtype=float).copy())
        if self.cost_max.shape != (3,) or not np.isfinite(self.cost_max).all() or np.any(self.cost_max < 0):
            raise ValueError("cost_max must contain three finite non-negative GLOBAL bounds")
        if np.any((self.cost_max_local > 0) & (self.cost_max == 0)):
            raise ValueError("cost_max must be positive for each applicable cost component")
        self.budget = None
        self.dqn_agent = None
        self.prb, self.power, self.aoi_norm = [], [], []
        self._has_reset = False

    def assign_agent(self, agent):
        self.dqn_agent = agent

    def reset(self, seed=None):
        """Reset traffic, resource history, estimator and all per-episode metrics."""
        if seed is not None:
            np.random.seed(seed)
        self.channel.rng = np.random.default_rng(seed)
        self.current_step = 0
        data = self.scenario.reset(seed=seed)
        self.bs_positions = data["bs_positions"]
        self.ue_positions = data["ue_positions"]
        self.target_states = data["target_states"]
        self.ue_mask = self.scenario.get_ue_mask().astype(bool)
        self.ue_indices = np.argwhere(self.ue_mask)
        self.aoi = np.zeros(self.num_targets, dtype=float)
        self.prob_suc = np.zeros(self.num_targets, dtype=float)
        self.detect_mask = np.zeros((self.num_bs, self.num_targets), dtype=bool)
        for b, measurement in enumerate(self.measurement_list):
            measurement.update_bs_pos(self.bs_positions[b])
        self.ekf_common.update_state(self._sample_initial_estimates(self.target_states, seed))
        self.pred_target_states = np.asarray(self.ekf_common.state_list).copy()
        self.last_u_pos, self.last_u_vel = self._get_uncertainty_summary()
        self.numBit.fill(0)
        self.URLLC_Latency.fill(0)
        self.eMBB_Thr.fill(0)
        self.BWP_slice.fill(0)
        self.last_BWP_slice.fill(0)
        self.budget = None
        self.p_res = np.full(self.num_bs, self.p_max, dtype=float)
        self.n_res = np.full(self.num_bs, self.total_prb, dtype=float)
        self.last_h_comm = np.zeros((self.num_bs, self.num_ue), dtype=float)
        self.last_h_sens = np.zeros((self.num_bs, self.num_targets), dtype=float)
        self.cost_by_bs = np.zeros((self.num_bs, self.cost_max.size), dtype=float)
        self.X_c = np.zeros((self.num_bs, len(self.bwp), self.num_slices, self.ue_mask.shape[1], self.max_prb))
        self.X_s = np.zeros((self.num_bs, len(self.bwp), self.num_targets, self.max_prb))
        self.served_ues = np.zeros(self.ue_mask.shape, dtype=bool)
        self.qos_served_ues = np.zeros(self.ue_mask.shape, dtype=bool)
        self.sinr_comm = np.zeros_like(self.X_c)
        self.sinr_sensing_prb = np.zeros_like(self.X_s)
        self.se_comm_mean = np.zeros((self.num_bs, *self.ue_mask.shape))
        self.se_comm_min = np.zeros_like(self.se_comm_mean)
        self.se_sensing_mean = np.zeros((self.num_bs, self.num_targets))
        self.se_sensing_min = np.zeros_like(self.se_sensing_mean)
        self.sinr_comm_mean = np.zeros_like(self.se_comm_mean)
        self.sinr_comm_min = np.zeros_like(self.se_comm_min)
        self.sinr_sensing_mean = np.zeros_like(self.se_sensing_mean)
        self.sinr_sensing_min = np.zeros_like(self.se_sensing_min)
        self.prb.clear()
        self.power.clear()
        self.aoi_norm.clear()
        self._has_reset = True
        return self._get_local_observations(), self._get_global_state()

    def _sample_initial_estimates(self, true_states, seed=None):
        prior_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        errors = prior_rng.multivariate_normal(
            np.zeros(self.ekf_common.state_dim), self.ekf_common.p_diag_init,
            size=self.num_targets)
        return np.asarray(true_states) + errors

    def set_budget(self, budget):
        """Set integer per-step PRB caps [B,W,M+S]; None removes caps.

        Target columns precede communication slice columns. The upper-level trainer owns
        frame duration, fractional-to-integer conversion and when caps change.
        Invalid actions are rejected, never silently projected to another policy action.
        """
        if budget is None:
            self.budget = None
            return
        budget = np.asarray(budget)
        if (budget.shape != self.BWP_slice.shape or
                not np.issubdtype(budget.dtype, np.integer) or np.any(budget < 0)
                or np.any(budget.sum(axis=2) > np.asarray(self.nPrb)[None, :])):
            raise ValueError("budget must be integer [B,W,M+S], with each BWP total <= nPrb")
        if not self.bwp_mode and np.any(np.count_nonzero(budget, axis=1) > 1):
            raise ValueError("single-BWP permits at most one BWP per BS and slice/target")
        self.budget = budget.copy()

    def action_mapping(self, actions_prb):
        """Return X_c [B,W,S,Umax,Rmax], X_s [B,W,M,Rmax]."""
        if not self._has_reset:
            raise RuntimeError("reset() is required before allocating resources")
        if hasattr(actions_prb, "detach"):
            actions_prb = actions_prb.detach().cpu().numpy()
        actions = np.asarray(actions_prb)
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("PRB destinations must have an integer dtype")
        if actions.shape == (self.num_bs, self.total_prb):
            blocks = np.zeros((self.num_bs, len(self.bwp), self.max_prb), dtype=actions.dtype)
            offset = 0
            for w, count in enumerate(self.nPrb):
                blocks[:, w, :count] = actions[:, offset:offset + count]
                offset += count
            actions = blocks
        expected = (self.num_bs, len(self.bwp), self.max_prb)
        if actions.shape != expected or np.any(actions < 0) or np.any(actions > self.num_targets + self.num_ue):
            raise ValueError(f"PRB action must have shape {expected} and destinations 0..{self.num_targets + self.num_ue}")
        if np.any(actions[:, ~self.prb_mask] != 0):
            raise ValueError("padded PRBs must be idle")
        x_c = np.zeros((self.num_bs, len(self.bwp), self.num_slices,
                        self.ue_mask.shape[1], self.max_prb), dtype=np.float32)
        x_s = np.zeros((self.num_bs, len(self.bwp), self.num_targets, self.max_prb), dtype=np.float32)
        for m in range(self.num_targets):
            x_s[:, :, m, :] = actions == m + 1
        for k, (s, u) in enumerate(self.ue_indices):
            x_c[:, :, s, u, :] = actions == self.num_targets + k + 1
        counts = np.concatenate((x_s.sum(axis=3), x_c.sum(axis=(3, 4))), axis=2)
        if not self.bwp_mode and np.any(np.count_nonzero(counts, axis=1) > 1):
            raise ValueError("single-BWP permits at most one BWP per BS and slice/target")
        if self.budget is not None and np.any(counts > self.budget):
            raise ValueError("allocation exceeds the current integer PRB budget")
        used = (actions > 0).sum(axis=2)
        self.n_res = self.total_prb - used.sum(axis=1)
        self.p_res = np.maximum(0., self.p_max - used @ self.pEach)
        return x_c, x_s

    def _reward_components(self, aoi, u_pos, u_vel, normalized_cost, service_ratios, *, frame=False):
        """Positive benefits to maximize; frame service ratios are flat per-slice."""
        ka, ku, ks = self.reward_k
        reduce_targets = np.mean if frame else np.sum
        sensing_aoi = float(reduce_targets(np.exp(-ka * np.asarray(aoi) / self.a_max)))
        sensing_uncertainty = float(reduce_targets(
            np.exp(-ku * np.asarray(u_pos) / self.u_pos_max)
            + np.exp(-ku * np.asarray(u_vel) / self.u_vel_max)))
        r_cost = float(3 - np.asarray(normalized_cost).sum())
        r_service = float(np.mean(1 - np.exp(-ks * np.asarray(service_ratios))))
        reward = float(self.alpha * sensing_aoi + self.beta * sensing_uncertainty
                       + self.gamma * r_cost + self.lambda_service * r_service)
        return reward, {"r_aoi": sensing_aoi, "r_uncertainty": sensing_uncertainty,
                        "r_cost": r_cost, "r_comm": r_service}

    def _compute_reward(self, user_rates, user_delay, cost, p_suc, u_pos, u_vel):
        """DQN-style team reward. Scheduling indicator differs from QoS success."""
        normalized_cost = np.divide(cost, self.cost_max, out=np.zeros_like(self.cost_max), where=self.cost_max > 0)
        pi = np.asarray(self.served_ues, dtype=float)
        allocation_ratio = float(pi[self.ue_mask].sum() / self.ue_mask.sum()) if self.ue_mask.any() else 0.
        reward, terms = self._reward_components(self.aoi, u_pos, u_vel, normalized_cost, [allocation_ratio])
        return reward, reward, 0., dict(terms, reward=reward, r_penalty=0.,
            norm_aoi=float(np.mean(self.aoi) / self.a_max), cost=np.asarray(cost).copy(),
            normalized_cost=normalized_cost, suc_embb=float(pi[:self.num_embb].sum() / (self.num_embb*self.num_ue_1embb)),
            suc_urllc=float(pi[self.num_embb:].sum() / (self.num_urllc*self.num_ue_1urllc)))

    def step(self, actions_prb):
        """Advance one physical step; return observations, scalar reward and diagnostics."""
        x_c, x_s = self.action_mapping(actions_prb)  # Validate before advancing time.
        self.current_step += 1
        self.last_BWP_slice = self.BWP_slice.copy()
        self.BWP_slice = np.concatenate((x_s.any(axis=3), x_c.any(axis=(3, 4))), axis=2).astype(np.int8)
        self.scenario.step_targets()
        self.target_states = self.scenario.get_target_states()
        for m in range(self.num_targets):
            self.ekf_common.predict(m)
        self.pred_target_states = np.asarray(self.ekf_common.state_list).copy()
        channel_out = self.channel.step(
            X_c=x_c, X_s=x_s, bs_pos=self.bs_positions, ue_pos=self.ue_positions,
            target_pos=self.target_states[:, :3], pred_target_pos=self.pred_target_states[:, :3])
        comm = self.comm_managers[0]
        self.sinr_comm = channel_out["comm_sinr_prb"].copy()
        spectral_eff = np.log2(1.0 + np.maximum(self.sinr_comm, 0.0))
        self.X_c, self.X_s = x_c.copy(), x_s.copy()
        self.served_ues = x_c.any(axis=(0, 1, 4))
        self.sinr_sensing_prb = channel_out["sensing_sinr_prb"].copy()
        self.sinr_comm_mean = self.sinr_comm.sum(axis=(1, 4)) / self.total_prb
        self.sinr_comm_min = np.where(self.prb_mask[None, :, None, None, :], self.sinr_comm, np.inf).min(axis=(1, 4))
        self.sinr_comm_min[:, ~self.ue_mask] = 0.
        self.sinr_sensing_mean = self.sinr_sensing_prb.sum(axis=(1, 3)) / self.total_prb
        self.sinr_sensing_min = np.where(self.prb_mask[None, :, None, :], self.sinr_sensing_prb, np.inf).min(axis=(1, 3))
        # Include idle but valid PRBs (zero SINR); exclude padding only.
        self.se_comm_mean = spectral_eff.sum(axis=(1, 4)) / self.total_prb
        self.se_comm_min = np.where(self.prb_mask[None, :, None, None, :], spectral_eff, np.inf).min(axis=(1, 4))
        self.se_comm_min[:, ~self.ue_mask] = 0.
        sensing_se = np.log2(1 + self.sinr_sensing_prb)
        self.se_sensing_mean = sensing_se.sum(axis=(1, 3)) / self.total_prb
        self.se_sensing_min = np.where(self.prb_mask[None, :, None, :], sensing_se, np.inf).min(axis=(1, 3))
        self.eMBB_Thr = comm.compute_embb_throughput(
            x_c[:, :, :self.num_embb, :self.num_ue_1embb],
            spectral_eff[:, :, :self.num_embb, :self.num_ue_1embb])
        current_bits = comm.compute_urllc_bits(
            x_c[:, :, self.num_embb:, :self.num_ue_1urllc],
            spectral_eff[:, :, self.num_embb:, :self.num_ue_1urllc], self.slot_duration)
        # Each UE has one packet in service. Excess bits are discarded;
        # the next packet starts at the following step.
        self.numBit += current_bits
        self.URLLC_Latency += self.slot_duration * 1000.
        latency_ms = self.URLLC_Latency.copy()
        completed = self.numBit >= self.required_bits_urllc
        completed_latency = np.where(completed, latency_ms, np.nan)
        # Frame service requires QoS, not merely a scheduled PRB.
        self.qos_served_ues.fill(False)
        self.qos_served_ues[:self.num_embb, :self.num_ue_1embb] = self.eMBB_Thr >= self.r_min * 1e6
        self.qos_served_ues[self.num_embb:, :self.num_ue_1urllc] = completed & (latency_ms <= self.max_ue_delay)
        self.qos_served_ues &= self.ue_mask
        urllc_scheduled = x_c[:, :, self.num_embb:, :self.num_ue_1urllc].any(axis=(0, 1, 4))
        required_bits_step = self.required_bits_urllc * self.slot_duration * 1000. / self.max_ue_delay
        self.numBit[completed] = 0.
        self.URLLC_Latency[completed] = 0.
        # Pool only real PRBs and real UEs for the existing observation layout.
        comm_features = spectral_eff.sum(axis=(1, 4)) / self.total_prb
        self.last_h_comm = comm_features[:, self.ue_mask]
        self.last_h_sens = np.log2(1 + channel_out["sensing_sinr_prb"]).sum(axis=(1, 3)) / self.total_prb
        self.prob_suc.fill(0.)
        self.detect_mask.fill(False)
        effective_sinr = channel_out["effective_sensing_sinr"]
        for m in range(self.num_targets):
            measurements, miss_probability = [], 1.
            for b in range(self.num_bs):
                if effective_sinr[b, m] <= 0:
                    continue
                detected, p_d = self.detection_list[b].detect(effective_sinr[b, m])
                miss_probability *= 1 - p_d
                self.detect_mask[b, m] = detected
                if detected:
                    measurement, covariance = self.measurement_list[b].gen_meas(
                        self.target_states[m], effective_sinr[b, m])
                    measurements.append((self.bs_positions[b], measurement, covariance))
            self.prob_suc[m] = 1 - miss_probability
            if measurements:
                self.ekf_common.update(m, measurements)
                self.aoi[m] = 0.
            else:
                self.aoi[m] += 1.
        u_pos, u_vel = self._get_uncertainty_summary()
        self.last_u_pos, self.last_u_vel = u_pos.copy(), u_vel.copy()
        self.cost_by_bs = self._costs_by_bs(x_c, x_s)
        cost = self.cost_by_bs.sum(axis=0)
        reward, _, _, reward_info = self._compute_reward(
            self.eMBB_Thr, latency_ms, cost, self.prob_suc, u_pos, u_vel)
        local_rewards = np.array([self._reward_components(
            self.aoi, u_pos, u_vel,
            np.divide(row, self.cost_max_local, out=np.zeros_like(self.cost_max_local), where=self.cost_max_local > 0),
            [float(self.served_ues[self.ue_mask].mean()) if self.ue_mask.any() else 0.])[0] for row in self.cost_by_bs])
        self.aoi_norm.append(float(np.mean(self.aoi) / self.a_max))
        self.prb.append(float(x_s.sum() / (self.num_bs * self.total_prb)))
        self.power.append(float(np.sum(x_s * self.pEach[None, :, None, None]) / (self.num_bs * self.p_max)))
        posterior = np.asarray(self.ekf_common.state_list)
        pos_error = np.linalg.norm(posterior[:, :3] - self.target_states[:, :3], axis=1)
        vel_error = np.linalg.norm(posterior[:, 3:] - self.target_states[:, 3:], axis=1)
        lost = bool(np.mean(u_pos > self.u_pos_max) > .5 or np.mean(u_vel > self.u_vel_max) > .5)
        embb_qos = self.qos_served_ues[:self.num_embb][self.ue_mask[:self.num_embb]]
        urllc_qos = self.qos_served_ues[self.num_embb:][self.ue_mask[self.num_embb:]]
        info = {
            "step": self.current_step, "time_seconds": self.current_step * self.slot_duration,
            "target_pos_mean_m": float(np.mean(np.linalg.norm(self.target_states[:, :3], axis=1))),
            "target_speed_mean_mps": float(np.mean(np.linalg.norm(self.target_states[:, 3:], axis=1))),
            "pos_error_mean_m": float(np.mean(pos_error)), "vel_error_mean_mps": float(np.mean(vel_error)),
            "aoi_mean": float(np.mean(self.aoi)), "u_pos_mean": float(np.mean(u_pos)),
            "u_vel_mean": float(np.mean(u_vel)), "lost_target": lost,
            "num_served_ues": float(np.count_nonzero(x_c.sum(axis=(0, 1, 4))[self.ue_mask])),
            "num_qos_ues": float(np.count_nonzero(self.qos_served_ues)),
            "qos_eMBB_rate": float(embb_qos.mean()) if embb_qos.size else 0.,
            "qos_URLLC_rate": float(urllc_qos.mean()) if urllc_qos.size else 0.,
            "mean_embb_rate_mbps": float(np.mean(self.eMBB_Thr) / 1e6),
            "comm_rate_mean_mbps": float((self.eMBB_Thr.sum() + current_bits.sum() / self.slot_duration) / (self.num_ue * 1e6)),
            "qos_satisfaction_rate": float((np.sum(self.eMBB_Thr >= self.r_min * 1e6) + np.sum(latency_ms <= self.max_ue_delay)) / self.num_ue),
            "mean_urllc_latency_ms": float(np.mean(latency_ms)),
            "urllc_bits_step": current_bits.copy(), "urllc_packet_completed": completed,
            "urllc_completed_latency_ms": completed_latency,
            "urllc_completed_on_time": completed & (latency_ms <= self.max_ue_delay),
            # Per-step service inequality is distinct from packet completion latency.
            "urllc_service_constraint_satisfied": current_bits >= required_bits_step * urllc_scheduled,
            "sensing_success_prob_mean": float(np.mean(self.prob_suc)),
            "energy_cost": float(cost[0]), "switch_cost": float(cost[1]),
            "guardband_cost": float(cost[2]),
            "cost_by_bs": self.cost_by_bs.copy(), "local_rewards": local_rewards}
        info.update(reward_info)
        return (self._get_local_observations(), self._get_global_state(), reward,
                lost, bool(self.current_step >= self.max_steps), info)

    def _get_uncertainty_summary(self):
        pos, vel = self.ekf_common.get_uncer()
        return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)

    def _get_local_uncertainty(self, bs_index):
        return self._get_uncertainty_summary()

    def _get_local_observations(self):
        """Build normalized per-BS observations from tracking and channel features."""
        u_pos, u_vel = self._get_uncertainty_summary()
        target_pos = self.pred_target_states[:, :3]
        t_frac = self.current_step / self.max_steps
        common = [np.clip(u_pos / self.u_pos_max, 0, 1),
                  np.clip(u_vel / self.u_vel_max, 0, 1),
                  np.clip(self.aoi / self.a_max, 0, 1),
                  np.clip(target_pos / self.d_max, -1, 1).ravel()]
        observations = []
        for b in range(self.num_bs):
            observations.append(np.concatenate(common + [
                np.clip(np.linalg.norm(target_pos - self.bs_positions[b], axis=1) / self.d_max, 0, 1),
                np.clip(self.last_h_comm[b] / self.se_comm_scale, 0, 1),
                np.clip(self.last_h_sens[b] / self.se_sens_scale, 0, 1),
                np.clip(self.bs_positions[b] / self.d_max, -1, 1),
                [self.p_res[b] / self.p_max, self.n_res[b] / self.total_prb,
                 t_frac, np.sin(2*np.pi*t_frac), np.cos(2*np.pi*t_frac)]]))
        return np.asarray(observations, dtype=np.float32)

    def _costs_by_bs(self, x_c, x_s):
        return np.column_stack((self._energy_by_bs(x_c, x_s),
                                self._switching_by_bs(), self._guardband_by_bs()))

    def _energy_by_bs(self, x_c, x_s):
        power = (x_c.sum(axis=(2, 3, 4)) + x_s.sum(axis=(2, 3))) @ self.pEach
        return power * self.dt / self.pa_efficiency

    def _switching_by_bs(self):
        # BWP_slice stores sensing targets before communication slices.
        delta = np.abs(self.BWP_slice.astype(float) - self.last_BWP_slice)
        return (self.cost_switch_s * delta[:, :, :self.num_targets].sum(axis=(1, 2))
                + self.cost_switch_c * delta[:, :, self.num_targets:].sum(axis=(1, 2)))

    def _guardband_by_bs(self):
        guard = np.zeros(self.num_bs)
        active = self.BWP_slice.any(axis=2)
        for w, other in self.adjacent_bwp_pairs:
            guard += (np.ceil(self.cost_gb * abs(self.bwp[w] - self.bwp[other]))
                      * active[:, w] * active[:, other])
        return guard

    def compute_energy_cost(self, X_c, X_s):
        """Total transmission energy in J, including power-amplifier efficiency."""
        return float(self._energy_by_bs(X_c, X_s).sum())

    def compute_switching_cost(self):
        """Charge both activation and deactivation for slices and targets."""
        return float(self._switching_by_bs().sum())

    def compute_guardband_cost(self):
        """Charge each active adjacent BWP pair once per BS."""
        return float(self._guardband_by_bs().sum())

    @abstractmethod
    def _get_global_state(self):
        """Return the global state vector for the selected environment."""
        raise NotImplementedError

    def plot_sensing_resource_usage(self, save_dir="results/figures"):
        """
        Vẽ và lưu hai biểu đồ:
        1. Tỷ lệ sử dụng PRB sensing theo aoi_norm.
        2. Tỷ lệ sử dụng công suất sensing theo aoi_norm.

        Các mẫu được gom vào từng khoảng aoi_norm nguyên:
        [0, 1), [1, 2), [2, 3), ...

        Lưu hình tại ``save_dir`` (mặc định ``results/figures``).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        save_dir = os.fspath(save_dir) if save_dir is not None else "results/figures"

        aoi_norm = np.asarray(self.aoi_norm, dtype=float).ravel()
        prb_sensing_ratio = np.asarray(self.prb, dtype=float).ravel()
        power_sensing_ratio = np.asarray(self.power, dtype=float).ravel()

        # Kiểm tra dữ liệu
        if not (
            len(aoi_norm)
            == len(prb_sensing_ratio)
            == len(power_sensing_ratio)
        ):
            raise ValueError(
                "aoi_norm, self.prb và self.power phải có cùng số phần tử."
            )

        if len(aoi_norm) == 0:
            raise ValueError("Dữ liệu đầu vào không được rỗng.")

        if not (
            np.all(np.isfinite(aoi_norm))
            and np.all(np.isfinite(prb_sensing_ratio))
            and np.all(np.isfinite(power_sensing_ratio))
        ):
            raise ValueError("Dữ liệu chứa NaN hoặc giá trị vô hạn.")

        if np.any(aoi_norm < 0):
            raise ValueError("aoi_norm không được âm.")

        # Gom nhóm theo khoảng AoI nguyên
        aoi_group = np.floor(aoi_norm).astype(int)

        x_min = int(np.min(aoi_group))
        x_max = int(np.max(aoi_group))
        x_bins = np.arange(x_min, x_max + 1)

        x_valid = []
        prb_mean = []
        power_mean = []

        for x_bin in x_bins:
            mask = aoi_group == x_bin

            if np.any(mask):
                x_valid.append(x_bin)
                prb_mean.append(
                    np.mean(prb_sensing_ratio[mask])
                )
                power_mean.append(
                    np.mean(power_sensing_ratio[mask])
                )

        x_valid = np.asarray(x_valid)
        prb_mean = np.asarray(prb_mean)
        power_mean = np.asarray(power_mean)

        # Tạo thư mục lưu hình
        os.makedirs(save_dir, exist_ok=True)

        # Hình 1: PRB sensing
        save_path = os.path.join(save_dir, "aoi_norm_prb.png")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            x_valid,
            prb_mean,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label="Sensing PRB ratio"
        )

        ax.set_xlabel("Normalized AoI ($AoI_{norm}$)", fontsize=12)
        ax.set_ylabel("Mean sensing PRB usage ratio", fontsize=12)
        ax.set_title("Sensing PRB Usage vs. Normalized AoI", fontsize=13)
        ax.set_xticks(x_bins)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"PRB figure saved to: {os.path.abspath(save_path)}")

        # Hình 2: Công suất sensing
        save_path = os.path.join(save_dir, "aoi_norm_power.png")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            x_valid,
            power_mean,
            marker="s",
            markersize=5,
            linewidth=1.8,
            label="Sensing power ratio"
        )

        ax.set_xlabel("Normalized AoI ($AoI_{norm}$)", fontsize=12)
        ax.set_ylabel("Mean sensing power usage ratio", fontsize=12)
        ax.set_title("Sensing Power Usage vs. Normalized AoI", fontsize=13)
        ax.set_xticks(x_bins)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Power figure saved to: {os.path.abspath(save_path)}")
