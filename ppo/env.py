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
        sim_config = sys_config["simulation"]
        req_config = sys_config["requirements"]
        scenario_config = sys_config["scenario"]

        self.num_bs = sys_config["num_bs"]
        self.num_ues = sys_config["num_ue"]
        self.num_targets = sys_config["num_target"]
        self.n_prb = sys_config["num_prb"]

        self.slot_duration = sys_config["slot_duration_ms"] / 1000.0  # Slot duration in seconds
        self.max_steps = sim_config["num_slots_per_episode"]
        
        # Power and scenario geometry limits
        self.p_max = float(sys_config["max_tx_power_watt"])
        self.noise_power = float(sys_config["noise_power_per_prb_watt"])
        self.d_max = float(scenario_config["diagonal_max_m"])

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
        self.u_pos_max = float(norm_cfg["u_pos_max"])                  
        self.u_vel_max = float(norm_cfg["u_vel_max"])              

        self.c1 = float(penalty_cfg["c1_rate_qos"])              
        self.c2 = float(penalty_cfg["c2_sensing_reliability"])   
        self.c3 = float(penalty_cfg["c3_pos_uncertainty"])
        self.c4 = float(penalty_cfg["c4_vel_uncertainty"])           

        self.current_step: int = 0
        self.aoi: np.ndarray = np.zeros(self.num_targets, dtype=np.float32)

        # Submodules initialization
        self.scenario = Scenario(self.config)
        self.channel = WirelessChannel(self.config)
        self.comm_managers = [
            CommunicationManager(self.config, bs_id=b) for b in range(self.num_bs)
        ]
        self.ekf = EKF(self.config)
        self.detection = Detection(self.config)
        self.measurement = BSMeasurement(self.config)

    def reset(self, seed=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            local_obs_list (np.ndarray): Local observations for all BSs [B, local_obs_dim]
            global_state (np.ndarray): Centralized state vector [global_state_dim]
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.aoi = np.zeros(self.num_targets, dtype=np.float32)
        self.bs_positions, self.ue_positions, self.target_states = self.scenario.reset()
        self.ekf.reset(self.target_states)
        self.pred_target_states = self.target_states.copy()

        # Initialize residual resources and channel state metrics
        self.p_res = np.full(self.num_bs, self.p_max, dtype=np.float32)
        self.n_res = np.full(self.num_bs, self.n_prb, dtype=np.float32)
        self.last_h_comm = np.zeros((self.num_bs, self.num_ues), dtype=np.float32)
        self.last_h_sens = np.zeros((self.num_bs, self.num_targets), dtype=np.float32)
        
        local_obs_list = self._get_local_observations()
        global_state = self._get_global_state()
        return local_obs_list, global_state

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
        """
        Compute reward R_t = R_obj - R_penalty:
        - R_obj: Minimize AoI, EKF tracking uncertainty (position + velocity), and total transmit power.
        - R_penalty: Hinge loss penalties for constraint violations:
            1. Communication QoS: UE rate < R_min
            2. Sensing reliability: Sensing success probability < P_min_sensing
            3. Target position uncertainty: EKF position uncertainty > U_pos_max
            4. Target velocity uncertainty: EKF velocity uncertainty > U_vel_max
            
        Returns:
            reward (float): Total scalar reward R_t
            r_obj (float): Primary objective value
            r_penalty (float): Total penalty value
            reward_info (dict): Detailed breakdown of reward terms
        """
        u_pos = np.asarray(u_pos, dtype=np.float32)
        u_vel = np.asarray(u_vel, dtype=np.float32)
        p_suc = np.asarray(p_suc, dtype=np.float32)
        user_rates = np.asarray(user_rates, dtype=np.float32)

        # 1. Primary objective R_obj
        norm_aoi = float(np.mean(self.aoi)) / self.a_max
        norm_uncertainty = (float(np.mean(u_pos)) / self.u_pos_max) + (float(np.mean(u_vel)) / self.u_vel_max)
        norm_power = float(np.sum(P_s) + np.sum(P_c)) / (self.num_bs * self.p_max)
        r_obj = - (self.alpha * norm_aoi + self.beta * norm_uncertainty + self.gamma * norm_power)

        # 2. Hinge loss penalty components
        user_rates_mbps = user_rates / 1e6
        pen_qos = self.c1 * float(np.sum(np.maximum(0.0, self.r_min - user_rates_mbps)) / self.r_min)
        pen_sensing = self.c2 * float(np.sum(np.maximum(0.0, self.p_min_sensing - p_suc)) / self.p_min_sensing)
        pen_upos = self.c3 * float(np.sum(np.maximum(0.0, u_pos - self.u_pos_max)) / self.u_pos_max)
        pen_uvel = self.c4 * float(np.sum(np.maximum(0.0, u_vel - self.u_vel_max)) / self.u_vel_max)

        r_penalty = pen_qos + pen_sensing + pen_upos + pen_uvel
        reward = float(r_obj - r_penalty)

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
        
        self.target_states = self.scenario.step_targets(self.slot_duration)
        target_pos = self.target_states[:, :3]  # True target coordinates (x, y, z)
    
        # 2. EKF predict
        self.pred_target_states = self.ekf.predict()  # Predicted target states s_hat(t|t-1)
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
        for b in range(self.num_bs):
            # Cắt lát ma trận của riêng BS b: shape (K, R)
            x_b = X_c[b]
            p_b = P_c[b]
            gain_b = comm_gain[b]
            # BS b tự tính SINR và throughput cục bộ
            sinr_b, se_b = self.comm_managers[b].compute_sinr(x_b, p_b, gain_b)
            rate_b = self.comm_managers[b].compute_user_rates(x_b, se_b)
            # Cộng dồn throughput các BS cấp cho UE (CoMP)
            user_rates += rate_b
            sinr_comm[b] = sinr_b
            # Cập nhật đặc trưng H_avg comm cục bộ của BS b [K]
            self.last_h_comm[b] = np.log2(1.0 + np.maximum(np.mean(sinr_b, axis=1), 0.0)).astype(np.float32)
        user_rates_mbps = user_rates / 1e6  # [K] (Mbps)
        # Update average spectral efficiency features H_avg cho sensing
        self.last_h_sens = np.log2(1.0 + np.maximum(np.mean(channel_out["sensing_sinr_prb"], axis=2), 0.0)).astype(np.float32)
        
        # 5. Sensing: Detection & measurement generation
        det_probs, p_suc = self.detection.compute_detection_probabilities(effective_sensing_sinr)
        
        # Sample binary detections D_{b,m} ~ Bernoulli(P_D)
        detection_mask = self.detection.sample_detections(det_probs)  # [B, M] bool
        
        # Collect measurements y_{b,m} = [r, theta, phi]^T and covariance matrices R_{b,m}
        measurements_dict, covariances_dict = self.measurement.get_measurements(
            bs_positions=self.bs_positions,
            target_positions=target_pos,
            effective_sinr=effective_sensing_sinr,
            detection_mask=detection_mask
        )
        
        # 6. EKF state update & AoI update
        for m in range(self.num_targets):
            if len(measurements_dict[m]) > 0:  # Target m detected by at least one BS
                self.ekf.batch_update(m, measurements_dict[m], self.bs_positions)
                self.aoi[m] = 0.0  # Reset AoI upon successful detection
            else:
                self.aoi[m] += 1.0  # Increment AoI when target is missed
                
        # 7. Compute reward (Objective + Hinge loss penalties)
        u_pos, u_vel = self.ekf.get_uncertainties()
        reward, r_obj, r_penalty, reward_info = self._compute_reward(
            P_s=P_s,
            P_c=P_c,
            user_rates=user_rates,
            p_suc=p_suc,
            u_pos=u_pos,
            u_vel=u_vel
        )          
        
        # 8. Check termination and truncation
        lost_target = bool(np.any(u_pos > self.u_pos_max) or np.any(u_vel > self.u_vel_max))
        terminated = bool(lost_target)
        truncated = bool(self.current_step >= self.max_steps)
        next_local_obs = self._get_local_observations()
        next_global_state = self._get_global_state()
        
        # Aggregate step statistics and diagnostics
        info = {
            "step": self.current_step,
            "aoi_mean": float(np.mean(self.aoi)),
            "u_pos_mean": float(np.mean(u_pos)),
            "u_vel_mean": float(np.mean(u_vel)),
            "comm_rate_mean_mbps": float(np.mean(user_rates_mbps)),
            "qos_satisfaction_rate": float(np.mean(user_rates_mbps >= self.r_min)),
            "sensing_success_prob_mean": float(np.mean(p_suc)),
            "lost_target": lost_target,
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
        
        Total dimension per BS: 7*M + K + 8.
        
        Returns:
            local_obs_list (np.ndarray): Shape [B, local_obs_dim]
        """
        u_pos, u_vel = self.ekf.get_uncertainties()
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

            # 2. Local communication and sensing channel features [K] and [M]
            h_comm_b = self.last_h_comm[b]  # [K]
            h_sens_b = self.last_h_sens[b]  # [M]

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
        u_pos, u_vel = self.ekf.get_uncertainties()
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

        # 3. Spectral efficiency features: [B * K] and [B * M]
        h_comm_flat = self.last_h_comm.flatten()
        h_sens_flat = self.last_h_sens.flatten()

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