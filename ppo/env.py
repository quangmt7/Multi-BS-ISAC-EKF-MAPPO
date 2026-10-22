import numpy as np
from scenario.scenario import Scenario
from channel.channel import WirelessChannel
from communication.communication import CommunicationManager
from sensing.detection import Detection
from sensing.bs_measurement import BSMeasurement
from sensing.ekf import EKF


class MultiISACEnv:
    """
    Multi-BS Integrated Sensing and Communication (ISAC) Environment.
    Coordinates wireless channel simulation, communication scheduling, sensing detection,
    EKF target tracking, and MAPPO reinforcement learning interface.
    """
    
    def __init__(self, config: dict):
        self.config = config

        # 1. System parameters
        sys_config = config["system_parameters"]
        ekf_config = config["ekf_parameters"]
        sim_config = sys_config["simulation"]
        req_config = sys_config["requirements"]
        scenario_config = sys_config["scenario"]

        self.num_bs = sys_config["num_bs"]
        self.num_ues = sys_config["num_ue"]
        self.num_targets = sys_config["num_target"]
        self.n_prb = sys_config["num_prb_bs"]

        self.slot_duration = sys_config["slot_duration_ms"] / 1000.0  # Slot duration in seconds
        self.max_steps = sim_config["num_slots_per_episode"]
        
        # Power and scenario geometry limits
        self.p_max = float(sys_config["max_tx_power_watt"])
        self.noise_power = float(sys_config["noise_power_per_prb_watt"])
        
        lx = float(scenario_config["area_size_x_m"])
        ly = float(scenario_config["area_size_y_m"])
        lz = float(scenario_config["area_size_z_m"])
        self.d_max = float(np.sqrt(lx**2 + ly**2 + lz**2))

        # System requirements
        self.r_min = float(req_config["min_ue_rate_mbps"])
        self.p_min_sensing = float(req_config["min_sensing_success_prob"])
        self.p_d_min = float(req_config["min_detection_prob"])
        self.p_fa = float(req_config["max_prob_false_alarm"])

        # 2. MAPPO parameters 
        mappo_cfg = config["mappo_parameters"]
        reward_cfg = mappo_cfg["reward_weights"]
        norm_cfg = mappo_cfg["normalization"]
        penalty_cfg = mappo_cfg["penalties"]
        
        self.alpha = float(reward_cfg["alpha_aoi"])                   
        self.beta = float(reward_cfg["beta_uncertainty"])              
        self.gamma = float(reward_cfg["gamma_power"])                 
      
        self.a_max = float(norm_cfg["a_max"])                         
        self.u_pos_max = float(ekf_config["uncertainty_limits"]["u_pos_max"])
        self.u_vel_max = float(ekf_config["uncertainty_limits"]["u_vel_max"])
        self.se_comm_scale = float(np.log2(1.0 + sys_config["sinr"]["gamma_c_min_linear"]))
        self.se_sens_scale = float(np.log2(1.0 + sys_config["sinr"]["gamma_s_min_linear"]))

        self.c1 = float(penalty_cfg["c1_rate_qos"])              
        self.c2 = float(penalty_cfg["c2_sensing_reliability"])   
        self.c3 = float(penalty_cfg["c3_pos_uncertainty"])
        self.c4 = float(penalty_cfg["c4_vel_uncertainty"])           

        self.current_step: int = 0
        self.aoi: np.ndarray = np.zeros(self.num_targets, dtype=np.float32)

        # 3. Submodules initialization
        self.scenario = Scenario(self.config)
        self.channel = WirelessChannel(self.config)
        self.comm_managers = [
            CommunicationManager(self.config, bs_id=b) for b in range(self.num_bs)
        ]
        self.ekf_config = ekf_config
        # For common EKF
        self.ekf_common = None
        self.detection_list = [Detection(self.p_fa) for _ in range(self.num_bs)]
        self.measurement_list = [
            BSMeasurement(b, self.ekf_config)
            for b in range(self.num_bs)
        ]

        # EKF common
        self.ekf_common = EKF(
            self.ekf_config["state_dim"], self.ekf_config["measurement_dim"],
            self.num_targets, self.ekf_config["dt"],
            self.ekf_config["process_noise"], 
            self.ekf_config["initial_covariance"], False
        )
        # EKF for each BS
        self.ekf_separate = []
        for _ in range(self.num_bs):
            self.ekf_separate.append(
            EKF(
            self.ekf_config["state_dim"], self.ekf_config["measurement_dim"],
            self.num_targets, self.ekf_config["dt"],
            self.ekf_config["process_noise"], 
            self.ekf_config["initial_covariance"], False
            ))
        
        # Initialize probability list
        self.prob_suc = np.zeros(self.num_targets)
        # Detection mask
        self.detect_mask = np.zeros((self.num_bs, self.num_targets))

    def reset(self, seed=None):
        """Reset the scenario and sample a reproducible EKF prior.

        Argument:
            (seed) (int | None): Optional episode seed.

        Return:
            (local_obs_list) (np.ndarray)[B, local_obs_dim]: Local BS observations.
            (global_state) (np.ndarray)[global_state_dim]: Centralized observation.

        Note:
            Initial estimation errors follow the configured P0; the true target
            states remain simulator data. Each reset restores EKF covariance.
        """
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self.channel, "rng"):
                self.channel.rng = np.random.default_rng(seed)
            
        self.current_step = 0
        self.aoi = np.zeros(self.num_targets, dtype=np.float32)

        # Scenario.reset() returns a dictionary of network positions
        scen_data = self.scenario.reset(seed=seed)
        self.bs_positions = scen_data["bs_positions"]
        self.ue_positions = scen_data["ue_positions"]
        self.target_states = scen_data["target_states"]

        self.ekf_common.update_state(self.target_states)
        for b in range(self.num_bs):
            self.measurement_list[b].update_bs_pos(self.bs_positions[b])
            self.ekf_separate[b].update_state(self.target_states)
        
        self.pred_target_states = self.target_states.copy()

        # Initialize residual resources and channel state metrics
        self.p_res = np.full(self.num_bs, self.p_max, dtype=np.float32)
        self.n_res = np.full(self.num_bs, self.n_prb, dtype=np.float32)
        self.last_h_comm = np.zeros((self.num_bs, self.num_ues), dtype=np.float32)
        self.last_h_sens = np.zeros((self.num_bs, self.num_targets), dtype=np.float32)

        local_obs_list = self._get_local_observations()
        global_state = self._get_global_state()
        # Reset the prob and detect_mask
        self.prob_suc = np.zeros(self.num_targets)
        self.detect_mask = np.zeros((self.num_bs, self.num_targets))
        return local_obs_list, global_state

    def _sample_initial_estimates(self, true_states, seed=None):
        """Sample target estimates with the configured initial covariance.

        Argument:
            (true_states) (np.ndarray)[N, state_dim]: Simulated initial states.
            (seed) (int | None): Optional episode seed for a separate prior stream.

        Return:
            (initial_estimates) (np.ndarray)[N, state_dim]: Estimates with
                independent zero-mean Gaussian errors distributed as P0.

        Note:
            This simulator initialization does not change the ground truth.
            Benchmark estimators should reuse copies of the sampled prior.
        """
        states = np.asarray(true_states, dtype=float)
        prior_seed = np.random.SeedSequence(seed).spawn(1)[0]
        prior_rng = np.random.default_rng(prior_seed)
        errors = prior_rng.multivariate_normal(
            np.zeros(self.ekf_common.state_dim),
            self.ekf_common.p_diag_init,
            size=states.shape[0],
        )
        return states + errors

    def action_mapping(self, actions_prb, actions_pwr):
        """
        Map policy action outputs to PRB allocations and transmit powers.
        
        Args:
            actions_prb: PRB destination choices [B, R], values in [0, M + K]
                         0: idle, 1..M: sensing targets, M+1..M+K: communication UEs
            actions_pwr: Normalized continuous power actions [B, R], values in [0, 1]
            
        Returns:
            X_s: Sensing PRB allocation indicator [B, M, R]
            X_c: Communication PRB allocation indicator [B, K, R]
            P_s: Sensing power allocation [B, M, R] (W)
            P_c: Communication power allocation [B, K, R] (W)
        """
        # Convert actions to numpy arrays if needed
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
        
        self.p_res = np.zeros(self.num_bs, dtype=np.float32)
        self.n_res = np.zeros(self.num_bs, dtype=np.float32)
        
        for b in range(self.num_bs):
            act_prb_b = actions_prb[b]
            raw_pwr_b = np.clip(actions_pwr[b], 0.0, 1.0)
            
            # Mask active PRBs (dest > 0: 0 is idle)
            active_mask = (act_prb_b > 0)
            self.n_res[b] = float(np.sum(~active_mask))  # Count idle PRBs
            
            pwr_allocated = np.zeros(self.n_prb, dtype=np.float32)
            if np.any(active_mask):
                # Normalize power across active PRBs only
                sum_active_pwr = float(np.sum(raw_pwr_b[active_mask]))
                if sum_active_pwr > 1.0:
                    pwr_allocated[active_mask] = (raw_pwr_b[active_mask] / sum_active_pwr) * self.p_max
                else:
                    pwr_allocated[active_mask] = raw_pwr_b[active_mask] * self.p_max        
            
            # Allocate PRBs and power to sensing (X_s, P_s) or communication (X_c, P_c)
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
                    
            # Compute residual transmit power for BS b
            used_power = float(np.sum(P_s[b]) + np.sum(P_c[b]))
            self.p_res[b] = max(0.0, self.p_max - used_power)
            
        return X_s, X_c, P_s, P_c

    def _compute_reward(self, P_s, P_c, user_rates, p_suc, u_pos, u_vel):
        """Compute the objective minus normalized constraint violations.

        Argument:
            (P_s) (array-like)[num_bs, num_target, num_prb]: Sensing powers in watts.
            (P_c) (array-like)[num_bs, num_ue, num_prb]: Communication powers in watts.
            (user_rates) (array-like)[num_ue]: User throughput in bits per second.
            (p_suc) (array-like)[num_target]: Sensing success probabilities.
            (u_pos) (array-like)[num_target]: Position uncertainty values.
            (u_vel) (array-like)[num_target]: Velocity uncertainty values.

        Return:
            (reward) (float): Total reward for the slot.
            (r_obj) (float): AoI, tracking uncertainty, and power objective.
            (r_penalty) (float): Non-negative weighted constraint violations.
            (reward_info) (dict[str, float]): Reward component values.

        Note:
            Each entity's shortfall is evaluated before averaging. Surplus
            throughput does not compensate for another UE's QoS violation.
        """
        u_pos = np.asarray(u_pos, dtype=np.float32)
        u_vel = np.asarray(u_vel, dtype=np.float32)
        p_suc = np.asarray(p_suc, dtype=np.float32)
        user_rates = np.asarray(user_rates, dtype=np.float32)
        user_rates_mbps = user_rates / 1e6

        # 1. Primary objective R_obj
        norm_aoi = float(np.mean(self.aoi)) / self.a_max
        norm_uncertainty = 0.5 * ((float(np.mean(u_pos)) / self.u_pos_max) + (float(np.mean(u_vel)) / self.u_vel_max))
        norm_power = float(np.sum(P_s) + np.sum(P_c)) / (self.num_bs * self.p_max)

        # =========================================================================
        # 2. QoS Truyền thông: CHỈ CHẶN SÀN, KHÔNG THƯỞNG THROUGHPUT
        # - Kẹp trần tại 1.0: Khi R_k >= R_min -> x1_qos = 1.0
        # - pen_qos = 0.0 khi đủ chuẩn; chỉ bị ÂM (PHẠT) khi R_k < R_min
        # =========================================================================
        x1_qos = float(np.mean(np.minimum(1.0, user_rates_mbps / self.r_min)))
        pen_qos = self.c1 * (x1_qos - 1.0)
        # =========================================================================
        # 3. Sensing P_suc: VỪA THƯỞNG VỪA PHẠT (KHÔNG KẸP TRẦN)
        # - Khi p_suc < 0.90 -> x2_sens < 1.0 -> pen_sensing ÂM (BỊ PHẠT)
        # - Khi p_suc = 0.90 -> x2_sens = 1.0 -> pen_sensing = 0.0 (MỐC CHUẨN)
        # - Khi p_suc > 0.90 -> x2_sens > 1.0 -> pen_sensing DƯƠNG (ĐƯỢC THƯỞNG!)
        # =========================================================================
        x2_sens = float(np.mean(p_suc / self.p_min_sensing))
        pen_sensing = self.c2 * (x2_sens - 1.0)
        # =========================================================================
        # 4. Bất định EKF: VỪA THƯỞNG VỪA PHẠT
        # - Khi u < u_max -> 1.0 - u / u_max > 0 -> pen_upos, pen_uvel DƯƠNG (ĐƯỢC THƯỞNG)
        # - Khi u = u_max -> 1.0 - u / u_max = 0 -> pen_upos, pen_uvel = 0.0 (MỐC CHUẨN)
        # - Khi u > u_max -> 1.0 - u / u_max < 0 -> pen_upos, pen_uvel ÂM (BỊ PHẠT)
        # =========================================================================
        pen_upos = self.c3 * (1.0 - float(np.mean(u_pos)) / self.u_pos_max)
        pen_uvel = self.c4 * (1.0 - float(np.mean(u_vel)) / self.u_vel_max)

        r_obj = - (self.alpha * norm_aoi + self.beta * norm_uncertainty + self.gamma * norm_power)
        r_penalty = pen_qos + pen_sensing + pen_upos + pen_uvel
        reward = float(r_obj + r_penalty)

        reward_info = {
            "reward": reward,
            "r_obj": r_obj,
            "r_penalty": r_penalty,
            "pen_qos": pen_qos,
            "pen_sensing": pen_sensing,
            "pen_upos": pen_upos,
            "pen_uvel": pen_uvel,
            "norm_aoi": norm_aoi,
            "norm_uncertainty": norm_uncertainty,
            "norm_power": norm_power,
            "violation_qos": float(np.mean(np.maximum(0.0, 1.0 - user_rates_mbps / self.r_min))),
            "violation_sensing": float(np.mean(np.maximum(0.0, 1.0 - p_suc / self.p_min_sensing))),
            "violation_pos": float(np.mean(np.maximum(0.0, u_pos / self.u_pos_max - 1.0))),
            "violation_vel": float(np.mean(np.maximum(0.0, u_vel / self.u_vel_max - 1.0))),
        }
        return reward, r_obj, r_penalty, reward_info

    def step(self, actions_prb, actions_pwr):
        """
        Execute one environment transition step.
        
        Args:
            actions_prb: PRB destination choices for each BS [B, R]
            actions_pwr: Continuous power allocation weights for each BS [B, R]
            
        Returns:
            next_local_obs (np.ndarray): Next local observations for all BSs [B, local_obs_dim]
            next_global_state (np.ndarray): Next centralized state vector [global_state_dim]
            reward (float): Step reward
            terminated (bool): Episode termination flag (target tracking lost)
            truncated (bool): Episode truncation flag (max steps reached)
            info (dict): Diagnostic and logging metrics
        """
        self.current_step += 1
        
        # 1. Action mapping
        X_s, X_c, P_s, P_c = self.action_mapping(actions_prb, actions_pwr)
        
        self.scenario.step_targets()
        self.target_states = self.scenario.get_target_states()
        target_pos = self.target_states[:, :3]  # True target coordinates (x, y, z)
    
        # 2. EKF predict (predict next target states for beamforming direction)
        for m in range(self.num_targets):
            self.ekf_common.predict(m)
            for b in range(self.num_bs):
                self.ekf_separate[b].predict(m)
        self.pred_target_states = np.asarray(self.ekf_common.state_list, dtype=np.float32)
        pred_target_pos = self.pred_target_states[:, :3]  # Predicted positions [M, 3]
        
        # 3. Channel simulation
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
        comm_gain = channel_out["comm_gain"]                          # [B, K, R]
        effective_sensing_sinr = channel_out["effective_sensing_sinr"]# [B, M]
        
        # 4. Communication: compute SINR, spectral efficiency, and user rates
        user_rates = np.zeros(self.num_ues, dtype=np.float32)
        sinr_comm = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)
        comm_power_by_bs = np.sum(P_c, axis=1)
        for b in range(self.num_bs):
            # Cắt lát ma trận của riêng BS b: shape (K, R)
            x_b = X_c[b]
            p_b = P_c[b]
            gain_b = comm_gain[b]
            # BS b tự tính SINR và throughput cục bộ
            other_bs_interference = self.channel.ibi_eta * np.sum(
                comm_power_by_bs[np.arange(self.num_bs) != b, np.newaxis, :]
                * comm_gain[np.arange(self.num_bs) != b], axis=0
            )
            sinr_b, se_b = self.comm_managers[b].compute_sinr(
                x_b, p_b, gain_b, inter_bs_interf=other_bs_interference
            )
            rate_b = self.comm_managers[b].compute_user_rates(x_b, se_b)
            # Cộng dồn throughput các BS cấp cho UE (CoMP)
            user_rates += rate_b
            sinr_comm[b] = sinr_b
            # H_avg comm of BS b [K] (average spectral efficiency of PRB)
            self.last_h_comm[b] = np.mean(se_b, axis=1).astype(np.float32)
        user_rates_mbps = user_rates / 1e6  # [K] (Mbps)
        # Update average spectral efficiency features H_avg cho sensing
        self.last_h_sens = np.mean(
            np.log2(1.0 + np.maximum(channel_out["sensing_sinr_prb"], 0.0)), axis=2
        ).astype(np.float32)
        
        # 5. Sensing: Each BS detect and build their detect maps
        self.prob_suc.fill(0.0)
        self.detect_mask.fill(0.0)
        for target in range(self.num_targets):
            miss_probability = 1.0
            for i in range(self.num_bs):
                if effective_sensing_sinr[i][target] > 0:
                    detected, p_d = self.detection_list[i].detect(effective_sensing_sinr[i][target])
                    miss_probability *= 1.0 - p_d
                    if detected == True:
                        self.detect_mask[i][target] = 1
            self.prob_suc[target] = 1.0 - miss_probability
        
        # Collect measurements y_{b,m} = [r, theta, phi]^T and covariance matrices R_{b,m}
        # Only measure if the detection is successfull
        measurements_by_target = [[] for _ in range(self.num_targets)]
        for target in range(self.num_targets):
            for i in range(self.num_bs):
                if self.detect_mask[i][target] == 1:
                    measurement, covariance = self.measurement_list[i].gen_meas(
                        target_state=self.target_states[target],
                        sensing_sinr=effective_sensing_sinr[i][target],
                    )
                    measurements_by_target[target].append(
                        (self.bs_positions[i], measurement, covariance)
                    )
        
        # 6. EKF state update & AoI update (both common and separate)
        for m in range(self.num_targets):
            if measurements_by_target[m]:
                self.ekf_common.update(m, measurements_by_target[m])
                self.aoi[m] = 0.0  # Reset AoI upon successful detection
            else:
                self.aoi[m] += 1.0  # Increment AoI when target is missed

            # Each-BS Local EKF: update ONLY with local measurements of BS b
            for b in range(self.num_bs):
                if self.detect_mask[b, m] == 1:
                    meas_b = [
                        (pos, meas, cov) for pos, meas, cov in measurements_by_target[m]
                        if np.array_equal(pos, self.bs_positions[b])
                    ]
                    if meas_b:
                        self.ekf_separate[b].update(m, meas_b)
                
        # 7. Compute reward (Objective + Hinge loss penalties)
        u_pos, u_vel = self.ekf_common.get_uncer()
        
        reward, _, _, reward_info = self._compute_reward(
            P_s=P_s,
            P_c=P_c,
            user_rates=user_rates,
            p_suc=self.prob_suc,
            u_pos=u_pos,
            u_vel=u_vel
        )          

        sep_pos_list = [self.ekf_separate[b].get_uncer()[0] for b in range(self.num_bs)]
        sep_vel_list = [self.ekf_separate[b].get_uncer()[1] for b in range(self.num_bs)]
        u_pos_best_bs = float(np.mean(np.min(sep_pos_list, axis=0)))
        u_vel_best_bs = float(np.mean(np.min(sep_vel_list, axis=0)))
        u_pos_mean_bs = float(np.mean(np.mean(sep_pos_list, axis=0)))
        u_vel_mean_bs = float(np.mean(np.mean(sep_vel_list, axis=0)))
        
        # 8. Check termination and truncation
        lost_target = bool(np.any(u_pos > self.u_pos_max) or np.any(u_vel > self.u_vel_max))
        terminated = bool(lost_target)
        truncated = bool(self.current_step >= self.max_steps)
        next_local_obs = self._get_local_observations()
        next_global_state = self._get_global_state()
        
        # Aggregate step statistics and diagnostics
        target_vel = self.target_states[:, 3:6]
        target_speeds = np.linalg.norm(target_vel, axis=1)
        pos_errors = np.linalg.norm(self.pred_target_states[:, :3] - target_pos, axis=1)
        vel_errors = np.linalg.norm(self.pred_target_states[:, 3:6] - target_vel, axis=1)
        num_served_ues = float(np.sum(np.sum(X_c, axis=(0, 2)) > 0))
        num_qos_ues = float(np.sum(user_rates_mbps >= self.r_min))

        info = {
            "step": self.current_step,
            "target_pos_mean_m": float(np.mean(np.linalg.norm(target_pos, axis=1))),
            "target_speed_mean_mps": float(np.mean(target_speeds)),
            "pos_error_mean_m": float(np.mean(pos_errors)),
            "vel_error_mean_mps": float(np.mean(vel_errors)),
            "num_served_ues": num_served_ues,
            "num_qos_ues": num_qos_ues,
            "aoi_mean": float(np.mean(self.aoi)),
            "u_pos_mean": float(np.mean(u_pos)),
            "u_vel_mean": float(np.mean(u_vel)),
            "comm_rate_mean_mbps": float(np.mean(user_rates_mbps)),
            "qos_satisfaction_rate": float(np.mean(user_rates_mbps >= self.r_min)),
            "sensing_success_prob_mean": float(np.mean(self.prob_suc)),
            "lost_target": lost_target,
            "u_pos_best_local_bs": u_pos_best_bs,
            "u_vel_best_local_bs": u_vel_best_bs,
            "u_pos_mean_local_bs": u_pos_mean_bs,
            "u_vel_mean_local_bs": u_vel_mean_bs,
        }
        info.update(reward_info)
        return next_local_obs, next_global_state, reward, terminated, truncated, info

    def _get_local_observations(self):
        """
        Construct local observation array o_b(t) for each BS b for the decentralized Actor.
        Feature components per BS b:
        - {U~_m^pos}:     [M] Target position uncertainties
        - {U~_m^vel}:     [M] Target velocity uncertainties
        - {A~_m}:         [M] Age of Information (AoI) for targets
        - {p^_m / d_max}: [3 * M] Normalized 3D predicted target coordinates
        - {d~_{b,m}^pred}:[M] Predicted relative distance from BS b to targets
        - {H~_{b,k}^avg}: [K] Communication channel spectral efficiency to UEs
        - {H~_{b,m}^avg}: [M] Sensing channel spectral efficiency to targets
        - {p_b / d_max}:  [3] Normalized 3D coordinates of BS b
        - P~_b^res:       [1] Residual power ratio of BS b
        - N~_b^res:       [1] Idle PRB ratio of BS b
        - progress:       [3] Time progress features (t / T, sin, cos)
        
        Total dimension per BS: 8*M + K + 8.
        
        Returns:
            local_obs_list (np.ndarray): Shape [B, local_obs_dim]
        """
        u_pos, u_vel = self.ekf_common.get_uncer()
        u_pos_norm = np.clip(u_pos / self.u_pos_max, 0.0, 1.0).astype(np.float32)  # [M]
        u_vel_norm = np.clip(u_vel / self.u_vel_max, 0.0, 1.0).astype(np.float32)  # [M]
        aoi_norm = np.clip(self.aoi / self.a_max, 0.0, 1.0).astype(np.float32)      # [M]

        # Normalized predicted target coordinates [3 * M]
        pred_target_pos = self.pred_target_states[:, :3]
        pred_target_pos_norm = np.clip(pred_target_pos / self.d_max, -1.0, 1.0).flatten().astype(np.float32)

        # Time progress features [3]
        t_frac = float(self.current_step) / float(max(1, self.max_steps))
        time_features = np.array([
            t_frac,
            np.sin(2.0 * np.pi * t_frac),
            np.cos(2.0 * np.pi * t_frac)
        ], dtype=np.float32)

        obs_list = []
        for b in range(self.num_bs):
            # 1. Predicted distance from BS b to all targets [M]
            d_pred_b = np.zeros(self.num_targets, dtype=np.float32)
            for m in range(self.num_targets):
                dist = float(np.linalg.norm(pred_target_pos[m] - self.bs_positions[b]))
                d_pred_b[m] = np.clip(dist / self.d_max, 0.0, 1.0)

            # 2. Local communication and sensing channel features [K] and [M] (normalized to [0, 1])
            h_comm_b = np.clip(self.last_h_comm[b] / self.se_comm_scale, 0.0, 1.0).astype(np.float32)
            h_sens_b = np.clip(self.last_h_sens[b] / self.se_sens_scale, 0.0, 1.0).astype(np.float32)

            # 3. Normalized 3D position of BS b [3]
            bs_pos_norm = np.clip(self.bs_positions[b] / self.d_max, -1.0, 1.0).astype(np.float32)

            # 4. Residual resources of BS b [1] and [1]
            p_res_b = np.array([np.clip(self.p_res[b] / self.p_max, 0.0, 1.0)], dtype=np.float32)
            n_res_b = np.array([np.clip(self.n_res[b] / self.n_prb, 0.0, 1.0)], dtype=np.float32)

            obs_b = np.concatenate([
                u_pos_norm,            # [M]
                u_vel_norm,            # [M]
                aoi_norm,              # [M]
                pred_target_pos_norm,  # [3 * M]
                d_pred_b,              # [M]
                h_comm_b,              # [K]
                h_sens_b,              # [M]
                bs_pos_norm,           # [3]
                p_res_b,               # [1]
                n_res_b,               # [1]
                time_features          # [3]
            ]).astype(np.float32)

            obs_list.append(obs_b)

        return np.stack(obs_list, axis=0)

    def _get_global_state(self):
        """
        Construct global state vector S_t for the Centralized Critic:
        S_t = [ {U~_m^pos}, {U~_m^vel}, {A~_m}, {d~_{b,m}^pred},
                {H~_{b,k}^avg}, {H~_{b,m}^avg}, {P~_b^res}, {N~_b^res} ]
                
        Total dimension: 3*M + 2*B*M + B*K + 2*B.
        
        Returns:
            global_state (np.ndarray): Shape [global_state_dim]
        """
        # 1. Uncertainty & AoI: [M], [M], [M]
        u_pos, u_vel = self.ekf_common.get_uncer()
        u_pos_norm = np.clip(u_pos / self.u_pos_max, 0.0, 1.0).astype(np.float32)
        u_vel_norm = np.clip(u_vel / self.u_vel_max, 0.0, 1.0).astype(np.float32)
        aoi_norm = np.clip(self.aoi / self.a_max, 0.0, 1.0).astype(np.float32)

        # 2. Predicted relative distances d~_{b,m}^pred: [B * M]
        pred_target_pos = self.pred_target_states[:, :3]
        d_pred = np.zeros((self.num_bs, self.num_targets), dtype=np.float32)
        for b in range(self.num_bs):
            for m in range(self.num_targets):
                dist = float(np.linalg.norm(pred_target_pos[m] - self.bs_positions[b]))
                d_pred[b, m] = np.clip(dist / self.d_max, 0.0, 1.0)
        d_pred_flat = d_pred.flatten()

        # 3. Spectral efficiency features normalized to [0, 1]: [B * K] and [B * M]
        h_comm_norm = np.clip(self.last_h_comm / self.se_comm_scale, 0.0, 1.0).astype(np.float32)
        h_sens_norm = np.clip(self.last_h_sens / self.se_sens_scale, 0.0, 1.0).astype(np.float32)
        h_comm_flat = h_comm_norm.flatten()
        h_sens_flat = h_sens_norm.flatten()

        # 4. Residual resources: [B] and [B]
        p_res_norm = np.clip(self.p_res / self.p_max, 0.0, 1.0).astype(np.float32)
        n_res_norm = np.clip(self.n_res / self.n_prb, 0.0, 1.0).astype(np.float32)

        global_state = np.concatenate([
            u_pos_norm,     # [M]
            u_vel_norm,     # [M]
            aoi_norm,       # [M]
            d_pred_flat,    # [B * M]
            h_comm_flat,    # [B * K]
            h_sens_flat,    # [B * M]
            p_res_norm,     # [B]
            n_res_norm      # [B]
        ]).astype(np.float32)

        return global_state
