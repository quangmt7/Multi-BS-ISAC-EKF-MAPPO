import numpy as np
import math
import os
from ppo.utils.data_analy import run_analysis
import matplotlib.pyplot as plt
from scenario.scenario import Scenario
from channel.channel import WirelessChannel
from communication.communication import CommunicationManager
from sensing.detection import Detection
from sensing.bs_measurement import BSMeasurement
from sensing.ekf import EKF
from abc import ABC, abstractmethod


class BaseISACEnv(ABC):
    """
    Base Multi-BS Integrated Sensing and Communication (ISAC) Environment.
    Coordinates wireless channel simulation, communication scheduling, sensing detection,
    EKF target tracking, and MAPPO reinforcement learning interface.
    This is base class for other environments for benchmark and framework 
    config : configuration
    cost_max : Cost max (for each type of cost) float[4]
    bwp_mode : True if this class is for Multi-BWP, False for Single-BWP
    """
    def __init__(self, config: dict, cost_max: list[float], bwp_mode: bool):
        self.config = config
        self.bwp_mode = bwp_mode

        # 1. System parameters
        sys_config = config["system_parameters"]
        ekf_config = config["ekf_parameters"]
        sim_config = sys_config["simulation"]
        req_config = sys_config["requirements"]
        scenario_config = sys_config["scenario"]

        self.num_bs = sys_config["num_bs"]
        self.num_embb = sys_config["num_embb"]
        self.num_urllc = sys_config["num_urllc"]
        self.num_slices = self.num_embb + self.num_urllc
        self.num_ue_1embb = sys_config["num_ue_1embb"]
        self.num_ue_1urllc = sys_config["num_ue_1urllc"]
        self.num_targets = sys_config["num_target"]
        self.num_ue = self.num_embb * self.num_ue_1embb + self.num_urllc * self.num_ue_1urllc

        self.slot_duration = sys_config["slot_duration_ms"] / 1000.0  # Slot duration in seconds
        self.max_steps = sim_config["num_slots_per_episode"]
        
        # Power and scenario geometry limits
        self.p_max = float(sys_config["max_tx_power_watt"])
        self.bwp = list(sys_config["bwp"])
        self.nPrb = list(sys_config["nPrb"])
        self.pEach = list(sys_config["pEach"])
        self.prbBw = float(sys_config["prbBw"])
        self.cost_gb = float(sys_config["cost_gb"])
        self.cost_switch = float(sys_config["cost_switch"])
        self.noise_power = float(sys_config["noise_power_per_prb_watt"])
        
        lx = float(scenario_config["area_size_x_m"])
        ly = float(scenario_config["area_size_y_m"])
        lz = float(scenario_config["area_size_z_m"])
        self.d_max = float(np.sqrt(lx**2 + ly**2 + lz**2))

        # System requirements
        self.r_min = float(req_config["min_ue_rate_mbps"])
        self.d_max = float(req_config["max_ue_delay"])
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
        self.required_bits_urllc = float(sys_config["urllc_packet_size_bits"])

        # Number of bits that an UE gained
        self.numBit = [[0 for _ in range(self.num_ue_1urllc)] for _ in range(self.num_urllc)]  
        
        # Latency for UEs of slice URLLC 
        self.URLLC_Latency = [[0 for _ in range(self.num_ue_1urllc)] for _ in range(self.num_urllc)]  

        # Throughput of UE eMBB
        self.eMBB_Thr = np.array([[0 for _ in range(self.num_ue_1embb)] for _ in range(self.num_embb)], np.float64)                    
      
        self.a_max = float(norm_cfg["a_max"])                         
        self.u_pos_max = float(ekf_config["uncertainty_limits"]["u_pos_max"])
        self.u_vel_max = float(ekf_config["uncertainty_limits"]["u_vel_max"])
        self.se_comm_scale = float(np.log2(1.0 + sys_config["sinr"]["gamma_c_min_linear"]))
        self.se_sens_scale = float(np.log2(1.0 + sys_config["sinr"]["gamma_s_min_linear"]))

        # Hệ số phạt vi phạm ràng buộc Hinge loss:
        # c1: Phạt QoS truyền thông (R_k < R_min)
        # c2: Phạt suy giảm xác suất sensing (P_m^suc < P_min)
        self.c1 = float(penalty_cfg.get("c1_rate_qos", penalty_cfg.get("c1", 0.2)))              
        self.c2 = float(penalty_cfg.get("c2_sensing_reliability", penalty_cfg.get("c2", 0.4)))   
        self.c3 = float(penalty_cfg.get("c3_pos_uncertainty", 0.0))
        self.c4 = float(penalty_cfg.get("c4_vel_uncertainty", 0.0))
        self.c5 = float(penalty_cfg.get("c5_sens_freq", 0.0))
        self.aoi_thr = float(penalty_cfg.get("aoi_thr", 0.0))
        self.delta_u_pos = float(penalty_cfg.get("delta_u_pos", 0.0))
        self.delta_u_vel = float(penalty_cfg.get("delta_u_vel", 0.0)) 
        self.own_ep = 0       

        self.current_step: int = 0
        self.aoi: np.ndarray = np.zeros(self.num_targets, dtype=np.float32)
        self.last_u_pos = np.zeros(self.num_targets, dtype=np.float32)
        self.last_u_vel = np.zeros(self.num_targets, dtype=np.float32)

        # 3. Submodules initialization
        self.scenario = Scenario(self.config)
        self.channel = WirelessChannel(self.config)
        self.comm_managers = [
            CommunicationManager(self.config, bs_id=b) for b in range(self.num_bs)
        ]
        self.ekf_config = ekf_config
        self.ekf_common = None
        self.ekf_separate = []
        self.detection_list = [Detection(self.p_fa) for _ in range(self.num_bs)]
        self.measurement_list = [
            BSMeasurement(b, self.ekf_config)
            for b in range(self.num_bs)
        ]

        self.ekf_common = EKF(
            self.ekf_config["state_dim"], self.ekf_config["measurement_dim"],
            self.num_targets, self.ekf_config["dt"],
            self.ekf_config["process_noise"], 
            self.ekf_config["initial_covariance"])

        # Initialize probability list
        self.prob_suc = np.zeros(self.num_targets)
        # Detection mask
        self.detect_mask = np.zeros((self.num_bs, self.num_targets))
        self.prb = []
        self.power = []
        self.aoi_norm = []

        # For BWP usage
        # Matrix shows if slice used BWP, with binary value
        self.BWP_slice = [[[0 for _ in range(self.num_slices + self.num_targets)] for _ in range(len(self.bwp))] \
                          for _ in range(self.num_bs)]
        self.last_BWP_slice = [[[0 for _ in range(self.num_slices + self.num_targets)] for _ in range(len(self.bwp))] \
                          for _ in range(self.num_bs)]
        self.cost_max = cost_max
        # DQN agent của RU
        self.dqn_agent = None

    def assign_agent(self, agent):
        """Gán DQN agent để dùng khi FrameEnv gọi env.select_action(), bắt buộc"""
        self.dqn_agent = agent

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
        self.last_u_pos = np.zeros(self.num_targets, dtype=np.float32)
        self.last_u_vel = np.zeros(self.num_targets, dtype=np.float32)

        # Scenario.reset() returns a dictionary of network positions
        scen_data = self.scenario.reset(seed=seed)
        self.bs_positions = scen_data["bs_positions"]
        self.ue_positions = scen_data["ue_positions"]
        self.target_states = scen_data["target_states"]

        for b in range(self.num_bs):
            self.measurement_list[b].update_bs_pos(self.bs_positions[b])

        if self.sepBS:
            for b in range(self.num_bs):
                self.ekf_separate[b].update_state(self.target_states)
        else:
            self.ekf_common.update_state(self.target_states)
        
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
        ekf_ref = self.ekf_common if self.sepBS else self.ekf_separate[0]
        prior_seed = np.random.SeedSequence(seed).spawn(1)[0]
        prior_rng = np.random.default_rng(prior_seed)
        errors = prior_rng.multivariate_normal(
            np.zeros(ekf_ref.state_dim),
            ekf_ref.p_diag_init,
            size=states.shape[0],
        )
        return states + errors

    def action_mapping(self, actions_prb):
        """
        Map policy action outputs to PRB allocations and transmit powers.
        
        Args:
            actions_prb: PRB destination choices [B, R], values in [0, M + K]
                        0: idle, 1..M: sensing targets, M+1..M+K: communication UEs
            
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

    def _compute_reward(self, user_rates, user_delay, cost, p_suc, u_pos, u_vel):
        """Compute the objective minus normalized constraint violations.
        Arguments:
            user_rates (np.ndarray): Achieved data rates per UE [num_embb][num_ue_1embb] (bps).
            user_delay (np.ndarray): Achieved delay per UE [num_urllc][num_ue_1urllc] (ms)
            cost (np.ndarray): Cost for BWP (energy, switching, fragmentation, guardband) [4]
            p_suc (np.ndarray): Detection / sensing success probabilities per target [num_targets].
            u_pos (np.ndarray): EKF position estimation uncertainty Tr(P_pos) per target [num_targets] (m^2).
            u_vel (np.ndarray): EKF velocity estimation uncertainty Tr(P_vel) per target [num_targets] (m^2/s^2).
        Returns:
            reward (float): Scalar reward returned to PPO agent (R_t).
            r_obj (float): Weighted objective value: alpha * R_sensing + (1 - alpha) * R_comm.
            r_penalty (float): Total non-negative penalty value (pen_qos + pen_sensing).
            reward_info (dict): Detailed diagnostic metrics for logging and TensorBoard:
                - 'reward': Total scalar reward (R_t).
                - 'r_sensing': Sensing objective score.
                - 'r_comm': Communication QoS satisfaction ratio.
                - 'r_penalty': Total penalty subtracted.
                - 'pen_sensing': Scaled penalty for sensing probability violations.
                - 'norm_aoi': Normalized AoI metric (A_bar / A_max).
                - 'cost': Cost for BWP (energy, switching, fragmentation, guardband),
                - 'suc_embb': sucess rate for embb,
                - 'suc_urllc': sucess rate for urllc,
                - 'violation_sensing': Average normalized sensing violation across targets.
        """
        # This is for temporary function to transform rate -> output used in reward
        k1 = 1
        k2 = 1
        #k3 = 1
        k4 = 1
        u_pos = np.asarray(u_pos, dtype=np.float32)
        u_vel = np.asarray(u_vel, dtype=np.float32)
        p_suc = np.asarray(p_suc, dtype=np.float32)
        user_rates = np.asarray(user_rates, dtype=np.float32)
        user_rates_mbps = user_rates / 1e6
        
        # 1. R_sensing 
        #norm_aoi = np.mean(self.aoi / self.a_max)
        a_bar = float(np.mean(np.exp(-k1*(self.aoi - self.aoi_thr / self.a_max))))
        u_pos_bar = float(np.mean(np.exp(-k2*(u_pos / self.u_pos_max))))
        u_vel_bar = float(np.mean(np.exp(-k2 * (u_vel / self.u_vel_max))))
        cost_rate = [cost[i] / self.cost_max[i] for i in range(4)]
        r_sensing = self.beta * a_bar + (1.0 - self.beta) * (u_pos_bar+u_vel_bar)
        
        # 2. R_communication
        # pi_e = 1 if R_k(t) >= R_k^min
        pi_e = (user_rates_mbps >= self.r_min).astype(np.float32)
        # pi_u = 1 if D_u(t) <= D_u^max
        pi_u = (user_delay <= self.d_max).astype(np.float32)
        r_comm = 1 - float(np.exp(-k4*((pi_e + pi_u) / self.num_ue)))

        # 3. R_penalty
        violation_aoi = self.c5 * np.mean(np.maximum(0.0, (self.aoi_thr - self.aoi) / self.a_max)**2)

        # pen_sensing
        violation_sensing = np.maximum(0.0, (self.p_min_sensing - p_suc) / self.p_min_sensing)
        pen_sensing = self.c2 * float(np.mean(violation_sensing**2))

        violation_pos = self.c3 * float(np.mean(np.maximum(
            0.0, (u_pos - self.u_pos_max) / self.u_pos_max)**2))

        violation_vel = self.c4 * float(np.mean(np.maximum(
            0.0, (u_vel - self.u_vel_max) / self.u_vel_max)**2))

        r_penalty = pen_sensing + violation_pos + violation_vel + violation_aoi

        # 4. Total reward
        r_obj = self.alpha * r_sensing + self.gamma * r_comm + (1 - self.alpha - self.gamma) * cost_rate
        reward = float(r_obj - r_penalty)

        reward_info = {
            "reward": reward,
            "r_sensing": r_sensing,
            "r_comm": r_comm,
            "r_penalty": r_penalty,
            #"pen_qos": pen_qos,
            "pen_sensing": pen_sensing,
            "norm_aoi": self.aoi / self.a_max,
            #"norm_power": norm_power,
            #"violation_qos": float(np.mean(violation_rate)),
            "cost": cost,
            "suc_embb": pi_e / self.num_ue_1embb * self.num_embb,
            "suc_urllc": pi_u / self.num_ue_1urllc * self.num_urllc,
            "violation_sensing": float(np.mean(violation_sensing)),
        }
        return reward, r_obj, r_penalty, reward_info

    def step(self, actions_prb):
        """
        Execute one environment transition step.
        
        Args:
            actions_prb: PRB destination choices for each BS [B, R]
        Returns:
            next_local_obs (np.ndarray): Next local observations for all BSs [B, local_obs_dim]
            next_global_state (np.ndarray): Next centralized state vector [global_state_dim]
            reward (float): Step reward
            terminated (bool): Episode termination flag (target tracking lost)
            truncated (bool): Episode truncation flag (max steps reached)
            info (dict): Diagnostic and logging metrics
        """
        self.current_step += 1
        self.own_ep += 1
        
        # 1. Action mapping
        X_s, X_c = self.action_mapping(actions_prb)
        
        self.scenario.step_targets()
        self.target_states = self.scenario.get_target_states()
        target_pos = self.target_states[:, :3]  # True target coordinates (x, y, z)
    
        # 2. EKF predict (predict next target states for beamforming direction)
        for m in range(self.num_targets):
            self.ekf_common.predict(m)
        self.pred_target_states = np.asarray(
            self.ekf_common.state_list, dtype=np.float32
        )
        pred_target_pos = self.pred_target_states[:, :3]
        
        # 3. Channel simulation
        channel_out = self.channel.step(
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
        # reset bits received in current slot
        current_bits = np.zeros(
            (self.num_urllc, self.num_ue_1urllc),
            dtype=np.float32
        )
        sinr_comm = []
        for b in range(self.num_bs):
            bs_se_list = []
            for w in range(len(self.bwp)):
                x_b_w = X_c[b][w]
                gain_b_w = comm_gain[b][w]
                sinr_b_w, se_b_w = self.comm_managers[b].compute_sinr(
                    x_b_w,
                    gain_b_w
                )
                sinr_comm.append(sinr_b_w)
                bs_se_list.append(se_b_w)

                # ==================================================
                # eMBB throughput
                # ==================================================
                for s in range(self.num_embb):
                    for e in range(self.num_ue_1embb):
                        self.eMBB_Thr[s][e] += (
                            self.comm_managers[b]
                            .compute_embb_rate(
                                slice_id=s,
                                ue_id=e,
                                bwp_id=w,
                                se=se_b_w
                            )
                        )

                # ==================================================
                # URLLC received bits
                # ==================================================

                for s in range(self.num_urllc):

                    for u in range(self.num_ue_1urllc):

                        current_bits[s][u] += (
                            self.comm_managers[b]
                            .compute_urllc_bits(
                                slice_id=s,
                                ue_id=u,
                                bwp_id=w,
                                se=se_b_w
                            )
                        )

            # ======================================================
            # H_avg communication feature
            # ======================================================
            if len(bs_se_list) > 0:
                se_concat = np.concatenate(
                    bs_se_list,
                    axis=-1
                )
                self.last_h_comm[b] = np.mean(
                    se_concat,
                    axis=1
                ).astype(np.float32)
        # ======================================================
        # Update URLLC queues
        # ======================================================
        for s in range(self.num_urllc):
            for u in range(self.num_ue_1urllc):
                self.numBit[s][u] += current_bits[s][u]
                # packet completed
                if self.numBit[s][u] >= self.required_bits_urllc:
                    self.URLLC_Latency[s][u] = 0.0
                    self.numBit[s][u] = 0.0
                else:
                    self.URLLC_Latency[s][u] += (
                        self.slot_duration * 1000.0
                    )
        # ======================================================
        # Statistics for logging
        # ======================================================
        embb_thr_mbps = (
            np.asarray(self.eMBB_Thr)
            / 1e6
        )
        urllc_latency_ms = np.asarray(
            self.URLLC_Latency,
            dtype=np.float32
        )
        # ======================================================
        # Sensing feature
        # ======================================================
        self.last_h_sens = np.mean(np.log2(1.0 + np.maximum(channel_out["sensing_sinr_prb"],0.0)),
            axis=2
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
        
        # 6. EKF state update & AoI update
        for m in range(self.num_targets):
            if not self.sepBS:
                # Shared EKF fuses all measurements available for target m.
                if measurements_by_target[m]:
                    self.ekf_common.update(m, measurements_by_target[m])
                    self.aoi[m] = 0.0
                else:
                    self.aoi[m] += 1.0
            else:
                # Each local EKF receives only measurements from its own BS.
                any_detection = False
                for b in range(self.num_bs):
                    if self.detect_mask[b, m] == 1:
                        meas_b = [
                            (pos, meas, cov)
                            for pos, meas, cov in measurements_by_target[m]
                            if np.allclose(pos, self.bs_positions[b], rtol=0.0, atol=1e-6)
                        ]
                        if meas_b:
                            self.ekf_separate[b].update(m, meas_b)
                            any_detection = True
                if any_detection:
                    self.aoi[m] = 0.0
                else:
                    self.aoi[m] += 1.0

        # 7. Compute reward from the active estimator mode.
        u_pos, u_vel = self._get_uncertainty_summary()

        # ==================================================
        # Cost calculation
        # ==================================================
        energy_cost = self.calculateEnergy(X_c=X_c, X_s=X_s)
        switch_cost = self.calculateSwitch()
        frag_cost = self.calculateFrag()
        guard_cost = self.calculateGuardBand()
        cost = np.array(
            [energy_cost, switch_cost, frag_cost, guard_cost],
            dtype=np.float32,
        )

        self.aoi_norm.append(np.mean(self.aoi) / self.a_max)

        reward, _, _, reward_info = self._compute_reward(
            user_rates=self.eMBB_Thr,
            user_delay=self.URLLC_Latency,
            cost=cost,
            p_suc=self.prob_suc,
            u_pos=u_pos,
            u_vel=u_vel
        )          

        # Save uncertainty for the next-step growth penalty.
        self.last_u_pos = np.asarray(u_pos, dtype=np.float32).copy()
        self.last_u_vel = np.asarray(u_vel, dtype=np.float32).copy()

        if not self.sepBS:
            u_pos_best_bs = float(np.mean(u_pos))
            u_vel_best_bs = float(np.mean(u_vel))
            u_pos_mean_bs = float(np.mean(u_pos))
            u_vel_mean_bs = float(np.mean(u_vel))
        else:
            sep_pos_list = np.stack(
                [ekf.get_uncer()[0] for ekf in self.ekf_separate], axis=0
            )
            sep_vel_list = np.stack(
                [ekf.get_uncer()[1] for ekf in self.ekf_separate], axis=0
            )
            u_pos_best_bs = float(np.mean(np.min(sep_pos_list, axis=0)))
            u_vel_best_bs = float(np.mean(np.min(sep_vel_list, axis=0)))
            u_pos_mean_bs = float(np.mean(sep_pos_list))
            u_vel_mean_bs = float(np.mean(sep_vel_list))
        
        # DEBUG
        #print(f"u_pos max: {np.max(u_pos)}, mean: {np.mean(u_pos)} lost: {np.mean(u_pos > self.u_pos_max)}\n"
        #      f"u_vel max: {np.max(u_vel)}, mean: {np.mean(u_vel)} lost: {np.mean(u_vel > self.u_vel_max)}\n")
        # 8. Check termination and truncation
        lost_target = bool(
            np.mean(u_pos > self.u_pos_max) > 0.5
            or
            np.mean(u_vel > self.u_vel_max) > 0.5
        )
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
        num_qos_ues = float(np.sum(self.eMBB_Thr >= self.r_min) + \
                            np.sum(self.URLLC_Latency <= self.d_max))

        if ((self.current_step / 100) >= 3500) and (self.current_step % 100 == 0):
            print("-----/-----\n")
            print(f"mean vel: {np.mean(vel_errors)} pos: {np.mean(pos_errors)} \n")
            print(f"median vel: {np.median(vel_errors)} pos: {np.median(pos_errors)} \n")
            print(f"std vel: {np.std(vel_errors)} pos: {np.std(pos_errors)} \n")
            print(f"p90 vel: {np.percentile(vel_errors, 0.9)} pos: {np.percentile(pos_errors, 0.9)} \n")
            print(f"p95 vel: {np.percentile(vel_errors, 0.95)} pos: {np.percentile(pos_errors, 0.95)} \n")
            print(f"max vel: {np.max(vel_errors)} pos: {np.max(pos_errors)} \n")
            print(f"suc vel: {np.count_nonzero(vel_errors > self.u_vel_max)} pos: {np.count_nonzero(pos_errors > self.u_pos_max)} \n")
                
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
            "qos_eMBB_rate": float(np.mean(self.eMBB_Thr >= self.r_min)),
            "qos_URLLC_rate": float(np.mean(self.URLLC_Latency <= self.d_max)),
            "sensing_success_prob_mean": float(np.mean(self.prob_suc)),
            "lost_target": lost_target,
            "u_pos_best_local_bs": u_pos_best_bs,
            "u_vel_best_local_bs": u_vel_best_bs,
            "u_pos_mean_local_bs": u_pos_mean_bs,
            "u_vel_mean_local_bs": u_vel_mean_bs,
            "energy_cost": float(energy_cost),
            "switch_cost": float(switch_cost),
            "fragmentation_cost": float(frag_cost),
            "guardband_cost": float(guard_cost),
        }
        info.update(reward_info)
        return next_local_obs, next_global_state, reward, terminated, truncated, info

    def _get_uncertainty_summary(self):
        """Return one [num_targets] uncertainty vector for reward/critic.

        In shared mode this is the common EKF output. In separate mode it is
        the mean across BS-local EKFs, keeping reward and critic dimensions fixed.
        """
        if not self.sepBS:
            u_pos, u_vel = self.ekf_common.get_uncer()
            return (np.asarray(u_pos, dtype=np.float32),
                    np.asarray(u_vel, dtype=np.float32))

        pos = np.stack([ekf.get_uncer()[0] for ekf in self.ekf_separate], axis=0)
        vel = np.stack([ekf.get_uncer()[1] for ekf in self.ekf_separate], axis=0)
        return pos.mean(axis=0).astype(np.float32), vel.mean(axis=0).astype(np.float32)

    def _get_local_uncertainty(self, bs_index):
        """Return uncertainty seen by one BS's local actor."""
        if self.sepBS:
            u_pos, u_vel = self.ekf_separate[bs_index].get_uncer()
        else:
            u_pos, u_vel = self.ekf_common.get_uncer()
        return np.asarray(u_pos, dtype=np.float32), np.asarray(u_vel, dtype=np.float32)

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
        aoi_norm = np.clip(self.aoi / self.a_max, 0.0, 1.0).astype(np.float32)

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
            u_pos_b, u_vel_b = self._get_local_uncertainty(b)
            u_pos_norm = np.clip(u_pos_b / self.u_pos_max, 0.0, 1.0).astype(np.float32)
            u_vel_norm = np.clip(u_vel_b / self.u_vel_max, 0.0, 1.0).astype(np.float32)

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

    def calculateEnergy(self, X_c, X_s):
        """
        Calculate energy cost
        Args:
            X_c: Allocation for comm UE (eMBB & URLLC) [b][w][s][u][r]
            X_p: Allocation for targets [b][w][m][r]
        Return:
            energy: Energy cost
        """
        energy = 0
        for b in range(self.num_bs):
            for w in range(len(self.bwp)):
                energy += (np.count_nonzero(X_c[b][w]) + np.count_nonzero(X_s[b][w])) * \
                           self.pEach[w] * self.slot_duration

        return energy
    
    def calculateSwitch(self):
        """
        Calculate BWP switching cost
        Args:
        Return:
            switch (float): switching cost 
        """
        switch = 0
        for b in range(self.num_bs):
            for w in range(len(self.bwp)):
                for s in range(self.num_slices + self.num_targets):
                    switch += self.cost_switch * self.BWP_slice[b][w][s] * (1 - self.last_BWP_slice[b][w][s])
        return switch
    
    def calculateFrag(self):
        """
        Calculate BWP fragmentation cost
        Args:
        Return:
            frag (float): fragmentation cost
        """
        frag = 0
        for b in range(self.num_bs):
            for s in range(self.num_slices + self.num_targets):
                frag += (np.sum(self.BWP_slice[b][w][s] for w in range(len(self.bwp))))**2
        return frag
    
    def calculateGuardBand(self):
        """
        Calculate BWP guardband cost
        Args:
        Return:
            guard (float): guardband cost
        """
        guard = 0
        for b in range(self.num_bs):
            for w in range(len(self.bwp)):
                for wk in range(len(self.bwp)):
                    if wk != w:
                        for s in range(self.num_slices + self.num_targets):
                            guard += self.cost_gb * np.abs(w - wk) * self.BWP_slice[b][w][s] * self.BWP_slice[b][wk][s]
        return guard

    @abstractmethod
    def _get_global_state(self):
        """
        Construct global state vector S_t for the Centralized Critic:
        
        Returns:
            global_state (np.ndarray): Shape [global_state_dim]
        """
        pass
    
    def plot_sensing_resource_usage(self):
        """
        Vẽ và lưu hai biểu đồ:
        1. Tỷ lệ sử dụng PRB sensing theo aoi_norm.
        2. Tỷ lệ sử dụng công suất sensing theo aoi_norm.

        Các mẫu được gom vào từng khoảng aoi_norm nguyên:
        [0, 1), [1, 2), [2, 3), ...

        Lưu hình tại results/figures/.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        save_dir = "results/figures"

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

        prb_results, power_results = run_analysis(
            self.aoi_norm,
            self.prb,
            self.power
        )

        #return {
        #    "aoi_bins": x_valid,
        #    "prb_mean": prb_mean,
        #    "power_mean": power_mean,
        #}
