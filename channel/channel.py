import numpy as np


class WirelessChannel:
      
    def __init__ (self, config):
         self.config= config
         sys_config= config.get("system_parameters")
         channel_config = sys_config.get("channel_and_interference")
         sim_config = sys_config.get("simulation")
         self.num_bs = sys_config["num_bs"]
         self.num_ues = sys_config["num_ue"]
         self.num_targets = sys_config["num_target"]
         self.n_prb = sys_config["num_prb"]
         
         self.carrier_freq = float(sys_config["carrier_frequency_ghz"]) * 1e9
         self.c = 3e8
         self.wavelength = self.c / self.carrier_freq
                
         # thema noise
         self.noise_power = float(sys_config["noise_power_per_prb_watt"])
         
         self.rcs = float(channel_config["target_rcs_sqm"])
         self.alpha_c = float(channel_config["pathloss_exponent_comm"])
         self.alpha_s = float(channel_config["pathloss_exponent_sensing"])
         self.g_tx = float(channel_config["antenna_gain_tx_linear"])
         self.g_rx = float(channel_config["antenna_gain_rx_linear"])
         self.beam_q= float(channel_config["beam_pattern_q"])
        # beamforming
         self.g_max = self.g_tx
         self.g_ant = float(channel_config["antenna_gain_rx_linear"])

  
         
        # Coupling factors
         self.c2s_eta0 = float(channel_config["c2s_leakage_eta0"])
         self.c2s_alpha = float(channel_config["c2s_leakage_alpha"])
         self.ibi_eta = float(channel_config["ibi_coupling_eta"])

        # Reference path loss at 1m: beta0 = (lambda / (4 * pi * 1))^2
         self.beta0 = float((self.wavelength / (4.0 * np.pi * 1.0)) ** 2)

        # Precompute C2S PRB distance leakage matrix eta_{r, r'} (R, R)
         prb_indices = np.arange(self.n_prb)
         prb_dist = np.abs(prb_indices[:, np.newaxis] - prb_indices[np.newaxis, :])
         self.c2s_matrix = (self.c2s_eta0 * np.exp(-self.c2s_alpha * prb_dist)).astype(np.float32)
        
        #rng
         seed = sim_config.get("seed", self.config.get("seed", 42))
         self.rng = np.random.default_rng(seed)
     
    def compute_comm_beamforming_gain(
        self, bs_pos, ue_pos, pred_ue_pos = None
    ) -> np.ndarray:
        """
        Compute communication beamforming gain G_{b,k}^{beam,c}(t) 
        Argument:
            bs_pos (numpy.ndarray)[B, 3]: 3D positions of Base Stations.
            ue_pos (numpy.ndarray)[K, 3]: True 3D positions of User Equipments p_k(t).
            pred_ue_pos (numpy.ndarray)[K, 3] : Predicted positions of UEs p^hat_k(t).

        Return:
            G_beam_c (numpy.ndarray)[B, K]: Communication beamforming gain matrix.
        """
        if pred_ue_pos is None:
            pred_ue_pos = ue_pos
        # 1.  u_{b,k}^{beam}(t) 
        diff_pred = pred_ue_pos[np.newaxis, :, :] - bs_pos[:, np.newaxis, :]
        dist_pred = np.maximum(np.sqrt(np.sum(diff_pred ** 2, axis=-1, keepdims=True)), 1e-6)
        u_beam = diff_pred / dist_pred

        # 2. real u_{b,k}(t) from BS to UE
        diff_true = ue_pos[np.newaxis, :, :] - bs_pos[:, np.newaxis, :]
        dist_true = np.maximum(np.sqrt(np.sum(diff_true ** 2, axis=-1, keepdims=True)), 1e-6)
        u_true = diff_true / dist_true

        # 3. Tích vô hướng (u_{b,k}^{beam})^T u_{b,k}: (B, K)
        inner_prod = np.sum(u_beam * u_true, axis=-1)
        inner_prod = np.clip(inner_prod, -1.0, 1.0)

        # 4.  G_max * |inner_prod|^q
        G_beam_c = self.g_max * (np.abs(inner_prod) ** self.beam_q)
        return G_beam_c.astype(np.float32)
 
     
     
     
    def compute_sensing_beamforming_gain(self, bs_pos, target_pos, pred_target_pos) :
        """
        Compute sensing beamforming gain
        Arguments:
             bs_pos (np.ndarray) [B,3] : 3D positions of BS
            target_pos (np.ndarray)[N, 3]: True 3D positions of targets or UEs.
            pred_target_pos (np.ndarray)[N, 3] : Predicted 3D positions from EKF.
         """
        if pred_target_pos is None:
          pred_target_pos = target_pos
        # u_{b,n}^{beam}
        diff_pred = pred_target_pos[np.newaxis, :, :] - bs_pos[:, np.newaxis, :]
        dist_pred = np.maximum(np.sqrt(np.sum(diff_pred ** 2, axis=-1, keepdims=True)), 1e-6)
        u_beam = diff_pred / dist_pred  # (B, N, 3)
  
        diff_true = target_pos[np.newaxis, :, :] - bs_pos[:, np.newaxis, :]
        dist_true = np.maximum(np.sqrt(np.sum(diff_true ** 2, axis=-1, keepdims=True)), 1e-6)
        u_true = diff_true / dist_true  # (B, N, 3)

        inner_prod = np.sum(u_beam * u_true, axis=-1)
        inner_prod = np.clip(inner_prod, -1.0, 1.0)

        # 4.
        G_beam_s = self.g_max * (np.abs(inner_prod) ** self.beam_q)
        return G_beam_s.astype(np.float32)



    
    def get_comm_channel_gain(self, bs_pos, ue_pos, pred_ue_pos=None) -> np.ndarray:
        """
        Compute effective communication channel gain G_{b,k,r}^c 
        Argument:
            (bs_pos) (numpy.ndarray)[B, 3]: 3D positions of Base Stations.
            (ue_pos) (numpy.ndarray)[K, 3]: 3D positions of User Equipments.
        Return:
            (comm_gain) (numpy.ndarray)[B, K, R]: Effective channel gain between BS and UE per PRB.
        """
        diff = bs_pos[:, np.newaxis, :] - ue_pos[np.newaxis, :, :]  # (B, K, 3)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (B, K)
        dist = np.maximum(dist,  1.0)
        
        # 1.Pathloss, shadowing
        pl_d0 = -10.0 * np.log10(self.beta0)
        sigma_sf = 8.0
        X_sigma = self.rng.normal(0.0, sigma_sf, size=(self.num_bs, self.num_ues))  # (B, K)
        pl_db = pl_d0 + 10.0 * self.alpha_c * np.log10(dist) + X_sigma              # Eq. (68)
        G_path = 10.0 ** (-pl_db / 10.0)                                             # Eq. (69)
        
        # 2. get beam gain
        beam_gain = self.compute_comm_beamforming_gain(bs_pos, ue_pos, pred_ue_pos)
        # 3.small scale fading
        h = (self.rng.standard_normal((self.num_bs, self.num_ues, self.n_prb))
     + 1j * self.rng.standard_normal((self.num_bs, self.num_ues, self.n_prb))) / np.sqrt(2.0)
        G_fading = np.abs(h) ** 2  # (B, K, R)
        # 4. comm gain
        comm_gain = (
            G_path[:, :, np.newaxis] 
            * beam_gain[:, :, np.newaxis]
            * self.g_ant
            * G_fading 
        )
        return comm_gain.astype(np.float32)
    
    def get_sensing_channel_gain(self, bs_pos, target_pos, pred_target_pos) -> np.ndarray:
        """
        Compute effective sensing channel gain G_{b,m}^s
        Argument:
            bs_pos (numpy.ndarray)[B, 3]: 3D positions of Base Stations.
            target_pos (numpy.ndarray)[M, 3]: True 3D positions of targets.
            pred_target_pos (numpy.ndarray)[M, 3]: Predicted target positions from EKF 
        Return:
            sensing_gain (numpy.ndarray)[B, M]: Effective radar channel gain.
        """
        diff = bs_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :]  # (B, M, 3)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (B, M)
        dist = np.maximum(dist, 1.0)

        # 1.Pathloss, shadowing
        pl_d0 = -10.0 * np.log10(self.beta0)
        sigma_sf = 8.0
        X_sigma = self.rng.normal(0.0, sigma_sf, size=(self.num_bs, self.num_targets))  # (B, M)
        pl_db = pl_d0 + 10.0 * self.alpha_s * np.log10(dist) + X_sigma                
        G_path = (10.0 ** (-pl_db / 10.0)) 
        # 2. get beam gain
        beam_gain = self.compute_sensing_beamforming_gain(bs_pos, target_pos, pred_target_pos)
        # 3.small scale fading
        h = (self.rng.standard_normal((self.num_bs, self.num_targets, self.n_prb))
     + 1j * self.rng.standard_normal((self.num_bs, self.num_targets, self.n_prb))) / np.sqrt(2.0)
        G_fading = np.abs(h) ** 2  # (B, K, R)
        # 6. calculate sensing gain
        sensing_gain = (
            G_path[:, :, np.newaxis]
            * self.g_ant
            * beam_gain[:, :, np.newaxis]
            * G_fading
        )
        return sensing_gain.astype(np.float32)

 
    def compute_c2s_interference(self, P_c: np.ndarray, comm_gain: np.ndarray) -> np.ndarray:
        """
        Compute Communication-to-Sensing (C2S) leakage interference 
        Return:
            c2s_interf (np.ndarray)[B, 1, R]: Leakage power per PRB r at BS b.
        """
        if P_c is None or comm_gain is None:
         return 0.0
        # total communication power per each PRB r' of base station b: (B, R)
        comm_power = np.sum(P_c * comm_gain, axis=1)
        c2s_interf = np.dot(comm_power, self.c2s_matrix)[:, np.newaxis, :]
        return c2s_interf 
 
 
 
    def compute_ibi_interference(self, X_s: np.ndarray, P_s: np.ndarray) -> np.ndarray:
        """
        Compute Inter-Base-Station Interference (IBI) 
        eta_IBI = 10^(-2) = 0.01
        """
        # 1. radar power  (B, M, R)
        p_radar = X_s * P_s
        # 2. Total radar power of base station: (B, R)
        p_bs = np.sum(p_radar, axis=1)
        # 3. total radar power of system : (1, R)
        p_total = np.sum(p_bs, axis=0, keepdims=True)
        # 4. other bs power (b' != b) (B, 1, R)
        other_bs_power = np.maximum(0.0, p_total - p_bs)[:, np.newaxis, :]
        # 5.
        ibi_interf = self.ibi_eta * other_bs_power
        return ibi_interf.astype(np.float32)
 

 
    def get_sensing_sinr(
        self,
        P_s: np.ndarray,
        X_s: np.ndarray,
        bs_pos: np.ndarray,
        target_pos: np.ndarray,
        pred_target_pos: np.ndarray = None,
        P_c: np.ndarray = None,
        comm_gain: np.ndarray = None,
    ) -> tuple:

        """
        Compute radar sensing SINR per PRB gamma_{b,m,r}^s and accumulated effective sensing SINR Gamma_{b,m}^s.
        Argument:
            P_s (numpy.ndarray)[B, M, R]: Radar sensing transmit power matrix in Watts.
            X_s (numpy.ndarray)[B, M, R]: Binary sensing PRB allocation matrix (1 if allocated, 0 otherwise).
            bs_pos (numpy.ndarray)[B, 3]: 3D positions of Base Stations [x, y, z] in meters.
            target_pos (numpy.ndarray)[M, 3]: True 3D positions of targets [x, y, z] in meters.
            pred_target_pos (numpy.ndarray )[M, 3]: Predicted 3D target positions from EKF for beam steering.
            P_c (numpy.ndarray )[B, K, R]: Communication transmit power matrix for C2S leakage calculation.
            comm_gain (numpy.ndarray )[B, K, R]: Effective communication channel gain matrix for C2S leakage.
        Return:
            effective_sensing_sinr (numpy.ndarray)[B, M]: Accumulated effective sensing SINR Gamma_{b,m}^s across PRBs.
            sensing_sinr_prb (numpy.ndarray)[B, M, R]: Instantaneous radar sensing SINR per PRB gamma_{b,m,r}^s.

        """

        # 1. G^s 
        sensing_gain = self.get_sensing_channel_gain(bs_pos, target_pos, pred_target_pos)
        signal_sensing = P_s * sensing_gain  # (B, M, R)

        c2s_interf = self.compute_c2s_interference(P_c, comm_gain)  
        ibi_interf = self.compute_ibi_interference(X_s, P_s)    
        total_interf = self.noise_power + c2s_interf + ibi_interf
        sensing_sinr_prb = np.where(
            (X_s == 1) & (P_s > 0),
            signal_sensing / np.maximum(total_interf, 1e-16),
            0.0
        ).astype(np.float32)
        # 5. Gamma_{b,m}^s 
        effective_sensing_sinr = np.sum(sensing_sinr_prb, axis=2).astype(np.float32)

        return effective_sensing_sinr, sensing_sinr_prb
    
    
    def step(
        self,
        P_c: np.ndarray,
        P_s: np.ndarray,
        X_c: np.ndarray,
        X_s: np.ndarray,
        bs_pos: np.ndarray,
        ue_pos: np.ndarray,
        target_pos: np.ndarray,
        pred_target_pos: np.ndarray = None,
        pred_ue_pos: np.ndarray = None,
    ) -> dict:
        """
        Simulate physical channel: compute comm channel gain and sensing SINR.
        """
        # 1. effective communication gain  G^c: (B, K, R)
        comm_gain = self.get_comm_channel_gain(bs_pos, ue_pos, pred_ue_pos)
        # 2. Sensing SINR radar 
        effective_sensing_sinr, sensing_sinr_prb = self.get_sensing_sinr(
            P_s=P_s,
            X_s=X_s,
            bs_pos=bs_pos,
            target_pos=target_pos,
            pred_target_pos=pred_target_pos,
            P_c=P_c,
            comm_gain=comm_gain
        )

        return {
            "comm_gain": comm_gain,                            # for communication.py tính Rate & QoS
            "effective_sensing_sinr": effective_sensing_sinr, # for detection.py & ekf.py
            "sensing_sinr_prb": sensing_sinr_prb,             # gamma_{b,m,r}^s
        }