import numpy as np
from scenario.scenario import Scenario
from channel.channel import WirelessChannel
from communication.communication import CommunicationManager
from sensing.detection import Detection
from sensing.bs_measurement import Measurement
from sensing.ekf import EKF  

class MultiISACEnv:
    
    def __init__(self, config: dict):
        self.config = config

        # get system parameter
        sys_config = config["system_parameters"]
        sim_config = sys_config["simulation"]
        req_config = sys_config["requirements"]

        self.num_bs = sys_config["num_bs"]
        self.num_ues = sys_config["num_ue"]
        self.num_targets = sys_config["num_target"]
        self.n_prb = sys_config["num_prb"]

        self.slot_duration = sys_config["slot_duration_ms"] / 1000.0  # 10.0 ms = 0.01 s
        self.max_steps = sim_config.get("num_slots_per_episode", 100)
        
        tx_dbm = sys_config["max_tx_power_dbm"]
        self.p_max = float(10.0 ** ((tx_dbm - 30.0) / 10.0))
        self.r_min = req_config["min_ue_rate_mbps"]

        # get mappo config
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
         
        self.c1 = float(penalty_cfg["penalty_rate_qos"])              
        self.c2 = float(penalty_cfg["penalty_sensing_reliability"])   
        self.c3 = float(penalty_cfg["penalty_uncertainty"])
        self.c4 = float(penalty_cfg["penalty_uncertainty"])           

      
        self.current_step: int = 0
        self.aoi: np.ndarray = np.zeros(self.num_targets, dtype=np.float32)

        # submodules
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