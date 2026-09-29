import numpy as np
from isac.scenario.scenario import Scenario
from isac.channel.channel import WirelessChannel
from isac.communication.communication import CommunicationManager
from isac.sensing.detection import Detection
from isac.sensing.measurement import Measurement
from isac.sensing.ekf import EKF  

class MultiISACEnv:
    
    def __init__(self, config: dict):
        self.config = config

        # get system parameter
        sys_cfg = config.get("system_parameters", config)
        sim_cfg = sys_cfg.get("simulation", {})
        req_cfg = sys_cfg.get("requirements", {})

        self.num_bs = sys_cfg["num_bs"]
        self.num_ues = sys_cfg["num_ue"]
        self.num_targets = sys_cfg["num_target"]
        self.n_prb = sys_cfg["num_prb"]

        # Thời lượng khe thời gian Delta t (s) và số slot mỗi episode T
        self.slot_duration = sys_cfg["slot_duration_ms"] / 1000.0  # 10.0 ms = 0.01 s
        self.max_steps = sim_cfg.get("num_slots_per_episode", 100)

        # Công suất phát cực đại: chuyển từ dBm sang Watt (33 dBm ~ 2.0 W)
        tx_dbm = sys_cfg.get("max_tx_power_dbm", 33.0)
        self.p_max = float(10.0 ** ((tx_dbm - 30.0) / 10.0))
        self.r_min = req_cfg.get("min_ue_rate_mbps", 1.0)

        # get mappo config
        mappo_cfg = config.get("mappo_parameters", config)
        reward_cfg = mappo_cfg.get("reward_weights", {})
        norm_cfg = mappo_cfg.get("normalization", {})

        self.alpha = reward_cfg.get("alpha_aoi", 0.4)
        self.beta = reward_cfg.get("beta_uncertainty", 0.4)
        self.gamma = reward_cfg.get("gamma_power", 0.2)

        self.a_max = norm_cfg.get("a_max", 20.0)
        self.u_pos_max = norm_cfg.get("u_pos_max", 20.0)
        self.u_vel_max = norm_cfg.get("u_vel_max", 10.0)

        # Biến theo dõi trong từng episode
        self.current_step: int = 0
        self.aoi: np.ndarray = np.zeros(self.num_targets, dtype=np.float32)

        # Khởi tạo các module con (submodules)
        self.scenario = Scenario(self.config)
        self.channel = WirelessChannel(self.config)
        self.comm = CommunicationManager(self.config)
        self.ekf = EKF(self.config)
        self.detection = Detection(self.config)
        self.measurement = Measurement(self.config)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.aoi = np.zeros(self.num_targets, dtype=np.float32)
        self.bs_positions, self.ue_positions, self.target_states = self.scenario.reset()
        self.ekf.reset(self.target_states)
        
        local_obs_list = self._get_local_observations()
        global_state = self._get_global_state()
        return local_obs_list, global_state

    def action_mapping(self, actions_prb, actions_pwr):
        X_s = np.zeros((self.num_bs, self.num_targets, self.n_prb), dtype=np.float32)
        X_c = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)
        P_s = np.zeros((self.num_bs, self.num_targets, self.n_prb), dtype=np.float32)
        P_c = np.zeros((self.num_bs, self.num_ues, self.n_prb), dtype=np.float32)
        
        for b in range(self.num_bs):
            raw_pwr = np.clip(actions_pwr[b], 0.0, 1.0)
            sum_pwr = float(np.sum(raw_pwr))
            # cal in percentages
            if sum_pwr > 1.0:
                pwr_allocated = (raw_pwr / sum_pwr) * self.p_max
            else:
                pwr_allocated = raw_pwr * self.p_max 

            for r in range(self.n_prb):
                dest = int(actions_prb[b, r])  # destination of the prb in b station
                val_power = pwr_allocated[r]

                if dest == 0:
                    continue
                elif 1 <= dest <= self.num_targets:
                    m = dest - 1
                    X_s[b, m, r] = 1.0  # binary variable
                    P_s[b, m, r] = val_power
                elif self.num_targets < dest <= self.num_targets + self.num_ues:
                    k = dest - self.num_targets - 1  # get number of ue
                    X_c[b, k, r] = 1.0
                    P_c[b, k, r] = val_power

        return X_s, X_c, P_s, P_c

    def step(self, actions_prb, actions_pwr):
        self.current_step += 1
        # 1. action mapping
        X_s, X_c, P_s, P_c = self.action_mapping(actions_prb, actions_pwr)
        self.target_states = self.scenario.step_targets(self.slot_duration)
        target_pos = self.target_states[:, :3]  # get position (x, y, z)
    
        # 2. Channel step
        
        
        # 3. communication compute rate UE and QoS penalty 
  

        # 4. sensing/detection.py to check if detect target


        # 5. Gọi sensing/measurement.py 


        # 6. Gọi sensing/ekf.py 
        self.ekf.predict()
        for m in range(self.num_targets):
            if len(measurements_dict[m]) > 0:  # D_m(t) = 1
                self.ekf.batch_update(m, measurements_dict[m], self.bs_positions)
                self.aoi[m] = 0.0  # new measurements -> Reset AoI
            else:
                self.aoi[m] += 1.0  # no new measurement
                
        # 7.  Reward 
        u_pos, u_vel = self.ekf.get_uncertainties()
        norm_aoi = float(np.mean(self.aoi)) / self.a_max
        norm_uncertainty = (float(np.mean(u_pos)) / self.u_pos_max) + (float(np.mean(u_vel)) / self.u_vel_max)
        total_pwr = float(np.sum(P_s) + np.sum(P_c))
        norm_power = total_pwr / (self.num_bs * self.p_max)

        reward = - (self.alpha * norm_aoi + self.beta * norm_uncertainty + self.gamma * norm_power) - comm_penalty
        
        # 8. 
        lost_target = bool(np.any(u_pos > self.u_pos_max) or np.any(u_vel > self.u_vel_max))
        terminated = bool(lost_target)
        truncated = bool(self.current_step >= self.max_steps)
        next_local_obs = self._get_local_observations()
        next_global_state = self._get_global_state()
        info = {
            "step": self.current_step,
            "aoi_mean": float(np.mean(self.aoi)),
            "u_pos_mean": float(np.mean(u_pos)),
            "u_vel_mean": float(np.mean(u_vel)),
            "comm_rate_mean": float(np.mean(comm_rates)),
            "qos_penalty": float(comm_penalty),
            "lost_target": lost_target,
        }
     return next_local_obs, next_global_state, reward, terminated, truncated, info

    def _get_local_observations(self):
         true

    def _get_global_state(self):
        true