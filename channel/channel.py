import numpy as np


class WirelessChannel:
      
    def __init__ (self, config):
         self.config= config
         sys_config= config.get("system_parameters")
         channel_config = sys_config.get("channel_and_interference")
         
         self.num_bs = sys_config["num_bs"]
         self.num_ue = sys_config["num_ue"]
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
         
         self.g_ant = float(channel_config.get("antenna_gain_rx_linear", 1.0))

  
         
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
        # u_{b,n}^{beam}
        diff_pred = pred_target_pos[np.newaxis, :, :] - bs_pos[:, np.newaxis, :]
        dist_pred = np.maximum(np.sqrt(np.sum(diff_pred ** 2, axis=-1, keepdims=True)), 1e-6)
        u_beam = diff_pred / dist_pred  # (B, N, 3)
  
        diff_true = target_pos[np.newaxis, :, :] - bs_pos[:, np.newaxis, :]
        dist_true = np.maximum(np.sqrt(np.sum(diff_true ** 2, axis=-1, keepdims=True)), 1e-6)
        u_true = diff_true / dist_true  # (B, N, 3)

        inner_prod = np.sum(u_beam * u_true, axis=-1)
        inner_prod = np.clip(inner_prod, -1.0, 1.0)

        # 4. Độ lợi búp sóng radar với hàm lũy thừa mũ q (Eq. 63): G_max * |inner_prod|^q
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
        
        # 1.Pathloss
        dist_km = dist/1000.0
        pl_db= 128.1 + 37.6* np.log10(dist_km)
        G_path = 10.0 **(-pl_db/10.0)
        
        # 2. get beam gain
        beam_gain = self.compute_comm_beamforming_gain(bs_pos, ue_pos, pred_ue_pos)
        # 3. small-scale fading
        rayleigh_fading= np.random.exponential(scale =1.0, size=(self.num_bs, self.num_ues, self.n_prb))
        # 4. shadowing
        sigma_sf = 8.0
        X_sigma =  np.random.normal(0.0, sigma_sf, size=(self.num_bs, self.num_ues))
        shadowing = 10.0 ** (X_sigma / 10.0)  # (B, K)
        # 5. comm gain
        comm_gain = (
            G_path[:, :, np.newaxis] 
            * beam_gain[:, :, np.newaxis]
            * self.g_ant
            * rayleigh_fading 
            * shadowing[:, :, np.newaxis] 
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

        # 1.Pathloss
        dist_km = dist/1000.0
        pl_db= 128.1 + 37.6* np.log10(dist_km)
        G_path = 10.0 **(-pl_db/10.0)

        # 2. G_{b,m}^{beam,s}(t)
        beam_gain = self.compute_sensing_beamforming_gain(bs_pos, target_pos, pred_target_pos)  # (B, M)

        # 3. small-scale fading
        rayleigh_fading= np.random.exponential(scale =1.0, size=(self.num_bs, self.num_ues, self.n_prb))
        # 4. shadowing
        sigma_sf = 8.0
        X_sigma =  np.random.normal(0.0, sigma_sf, size=(self.num_bs, self.num_ues))
        shadowing = 10.0 ** (X_sigma / 10.0)  # (B, K)
        # 5.
        sensing_gain = (
            G_path[:, :, np.newaxis]
            * self.g_ant
            * beam_gain[:, :, np.newaxis]
            * rayleigh_fading
            * shadowing[:, :, np.newaxis]
        )
        return sensing_gain.astype(np.float32)

 
 
    def get_sensing_sinr(
        self,
        P_s: np.ndarray,
        X_s: np.ndarray,
        bs_pos: np.ndarray,
        target_pos: np.ndarray,
        pred_target_pos: np.ndarray,
        P_c: np.ndarray = None,
        comm_gain: np.ndarray = None,
    ) -> tuple:
        return effective_sensing_sinr, sensing_sinr_prb
