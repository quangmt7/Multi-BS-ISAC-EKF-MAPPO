import numpy as np


class WirelessChannel:
      
    def __init__ (self, config):
         self.config= config
         sys_config= config.get("system_parameters")
         channel_config = sys_config.get("channel_and_interference")
         sim_config = sys_config.get("simulation")
         self.bwp = sys_config["bwp"]
         self.num_targets = sys_config["num_target"]
         self.num_bs = sys_config["num_bs"]
         self.nPrb = sys_config["nPrb"]
         self.maxPrb = max(self.nPrb)
         self.prbBW = sys_config["prbBW"]
         self.shadowing = float(channel_config["shadowing_std_db"])
         
         self.carrier_freq = float(sys_config["carrier_frequency_ghz"]) * 1e9
         self.fStart = list(sys_config["fStart"])
         self.oleF = float(sys_config["oleF"])
         self.pEach = list(sys_config["pEach"])
         self.c = 3e8
         self.wavelength = self.c / self.carrier_freq
                
         # thema noise
         self.noise_power = float(sys_config["noise_power_per_prb_watt"])
         
         self.rcs = float(channel_config["target_rcs_sqm"])
         if not np.isfinite(self.rcs) or self.rcs < 0:
             raise ValueError("target_rcs_sqm must be finite and non-negative")
         self.alpha_c = float(channel_config["pathloss_exponent_comm"])
         self.alpha_s = float(channel_config["pathloss_exponent_sensing"])
         if not np.isclose(self.alpha_s, 2.0):
             raise ValueError("monostatic echo gain requires pathloss_exponent_sensing = 2")
         self.g_tx = float(channel_config["antenna_gain_tx_linear"])
         self.g_rx = float(channel_config["antenna_gain_rx_linear"])
         self.beam_q= float(channel_config["beam_pattern_q"])
        # beamforming
         self.g_max = self.g_tx
         self.g_ant = float(sys_config["ue_parameters"]["antenna_gain_rx_linear"])

        # Coupling factors
         self.c2s_eta0 = float(channel_config["c2s_leakage_eta0"])
         self.c2s_alpha = float(channel_config["c2s_leakage_alpha"])
         self.ibi_eta = float(channel_config["ibi_coupling_eta"])

        # Reference path loss at 1m: beta0 = (lambda / (4 * pi * 1))^2
        # This should be taken through config file
         self.pathloss_const_db = float(channel_config["pathloss_const_db"])

        # Precompute C2S PRB distance leakage matrix eta_{r, r'} (R, R)
        # Spectral leakage matrix Omega(delta_f), shape [W,R,W,R]
        # c2s_matrix[w_tx,r_tx,w_rx,r_rx]
        # represents spectral leakage from communication PRB (w_tx,r_tx) to sensing PRB (w_rx,r_rx).
         num_bwp = len(self.fStart)
         valid_prb = np.zeros((num_bwp, self.maxPrb), dtype=np.float32)

         for w in range(num_bwp):
            valid_prb[w, :self.nPrb[w]] = 1.0
         valid_pair = (valid_prb[:, :, None, None] * valid_prb[None, None, :, :])
         prb_freq = np.zeros((num_bwp, self.maxPrb), dtype=np.float32)

         for w in range(num_bwp):
            delta_f_w = 15e3 * (2 ** self.bwp[w])
            prb_idx = np.arange(self.nPrb[w], dtype=np.float32)
            prb_freq[w, :self.nPrb[w]] = (self.fStart[w] + 12.0 * delta_f_w * (prb_idx + 0.5))

         freq_tx = prb_freq[:, :, None, None]
         freq_rx = prb_freq[None, None, :, :]

         delta_freq = np.abs(freq_tx - freq_rx)

         prb_bw_w = (self.prbBW * (2 ** np.asarray(self.bwp, dtype=np.float32)))

         sigma_f = (self.oleF * prb_bw_w)
         sigma_f = sigma_f[:, None, None, None]

         self.c2s_matrix = (np.exp(-(delta_freq ** 2) / (2.0 * sigma_f ** 2)) * valid_pair).astype(np.float32)
        #rng
         seed = sim_config.get("seed", self.config.get("seed", 42))
         self.rng = np.random.default_rng(seed)
     
    def compute_comm_beamforming_gain(self, bs_pos, ue_pos, pred_ue_pos=None):
        """
        Compute communication beamforming gain.
        Parameters
            bs_pos : (np.ndarray, [B,3])
            ue_pos : (np.ndarray, [S,U,3])
            pred_ue_pos : (np.ndarray | None, [S,U,3])
        Returns
            G_beam_c : (np.ndarray, [B,S,U])
        """
        if pred_ue_pos is None:
            pred_ue_pos = ue_pos
        diff_pred = (pred_ue_pos[None,:,:,:] - bs_pos[:,None,None,:])
        dist_pred = np.maximum(np.linalg.norm(diff_pred, axis=-1, keepdims=True), 1e-6)
        u_beam = diff_pred / dist_pred

        diff_true = (ue_pos[None,:,:,:] - bs_pos[:,None,None,:])
        dist_true = np.maximum(np.linalg.norm(diff_true, axis=-1, keepdims=True), 1e-6)
        u_true = diff_true / dist_true

        inner_prod = np.sum(u_beam * u_true, axis=-1)
        inner_prod = np.clip(inner_prod, -1.0, 1.0)
        G_beam_c = (self.g_max * np.abs(inner_prod) ** self.beam_q)

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


    def get_comm_channel_gain(self, bs_pos, ue_pos,
                          pred_ue_pos=None):
        """
        Compute effective communication channel gain.
        Parameters
            bs_pos : (np.ndarray: [B, 3]) BS coordinates.
            ue_pos : (np.ndarray: [S, U, 3]) UE coordinates.
            pred_ue_pos : (np.ndarray | None: [S, U, 3]) Predicted UE positions used for beam steering.
                        If None, true positions are used.
        Returns
            comm_gain : (np.ndarray, [B, W, S, U, R] Effective communication channel gain.
        """

        if pred_ue_pos is None:
            pred_ue_pos = ue_pos
        num_bwp = len(self.fStart)
        # Distance BS -> UE
        diff = (bs_pos[:, None, None, :] - ue_pos[None, :, :, :])
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.maximum(dist, 1.0)
        # Pathloss + shadowing
        X_sigma = self.rng.normal(0.0, self.shadowing, size=(self.num_bs, ue_pos.shape[0], ue_pos.shape[1]))
        pl_db = (self.pathloss_const_db + self.alpha_c * np.log10(dist / 1000.0) + X_sigma)
        G_path = 10.0 ** (-pl_db / 10.0)
        # Beamforming gain

        diff_pred = (pred_ue_pos[None, :, :, :] - bs_pos[:, None, None, :])
        dist_pred = np.maximum(np.linalg.norm(diff_pred, axis=-1, keepdims=True), 1e-6)
        u_beam = diff_pred / dist_pred
        diff_true = (ue_pos[None, :, :, :] - bs_pos[:, None, None, :])
        dist_true = np.maximum(np.linalg.norm(diff_true, axis=-1, keepdims=True), 1e-6)

        u_true = diff_true / dist_true
        inner_prod = np.sum(u_beam * u_true, axis=-1)

        inner_prod = np.clip(inner_prod, -1.0, 1.0)
        beam_gain = (self.g_max * np.abs(inner_prod) ** self.beam_q)

        # Small-scale fading

        h = (self.rng.standard_normal((self.num_bs, len(self.bwp), ue_pos.shape[0], ue_pos.shape[1], self.maxPrb))
            + 1j * self.rng.standard_normal((self.num_bs, len(self.bwp), ue_pos.shape[0], ue_pos.shape[1], self.maxPrb))
        ) / np.sqrt(2.0)
        G_fading = np.abs(h) ** 2

        # Effective gain
        base_gain = (G_path[..., None] * beam_gain[..., None] * self.g_ant * G_fading)

        return base_gain.astype(np.float32)
    
    def get_sensing_channel_gain(self, bs_pos, target_pos, pred_target_pos):
        """
        Compute effective sensing channel gain.
        Parameters
            bs_pos : (np.ndarray, [B,3]) BS coordinates.
            target_pos : (np.ndarray, [M,3]) True target positions.
            pred_target_pos : (np.ndarray, [M,3]) EKF predicted target positions used for beam steering.
        Returns
            sensing_gain : (np.ndarray, [B,W,M,R]) Effective sensing channel gain.
        """
        num_bwp = len(self.fStart)

        # BS -> Target distance
        diff = (bs_pos[:, None, :] - target_pos[None, :, :])
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.maximum(dist, 1.0)

        # Radar path gain
        X_sigma = self.rng.normal(0.0, self.shadowing, size=(self.num_bs, self.num_targets))
        G_path = (self.wavelength ** 2 * self.rcs / ((4.0 * np.pi) ** 3 * dist ** 4) * \
            10.0 ** (-2.0 * X_sigma / 10.0))

        # Beamforming gain
        beam_gain = (self.compute_sensing_beamforming_gain(bs_pos, target_pos, pred_target_pos))

        # Small-scale fading
        h = (self.rng.standard_normal((self.num_bs, len(self.bwp), self.num_targets, self.maxPrb)) + \
            1j * self.rng.standard_normal((self.num_bs, len(self.bwp), self.num_targets, self.maxPrb))) / np.sqrt(2.0)
        G_fading = np.abs(h) ** 2
        
        # Effective sensing gain
        base_gain = (G_path[:, :, None] * beam_gain[:, :, None] ** 2 * G_fading)

        # Expand BWP dimension
        sensing_gain = np.repeat(base_gain[:, None, :, :], num_bwp, axis=1)

        return sensing_gain.astype(np.float32)

 
    def compute_c2s_interference(self, X_c):
        """
        Compute communication-to-sensing leakage.
        Parameters
            X_c : (np.ndarray, [B,W,S,U,R]) Communication allocation.
        Returns
            c2s_interf : (np.ndarray, [B,W,M,R]) Communication leakage seen by sensing receivers.
        """
        if X_c is None:
            return 0.0

        # Communication transmit power, shape [B,W,S,U,R]
        p_prb = np.asarray(self.pEach, dtype=np.float32)[None,:,None,None,None]
        # Delete the comm_gain here, the gain from Tx communication -> Rx receiver
        comm_power = (X_c * self.c2s_eta0 * p_prb)

        # Sum over slice and UE, shape [B,W,R]
        comm_power = np.sum(comm_power, axis=(2,3))

        # Spectral leakage c2s_matrix: [W,R,W,R]. Result [B,W,R]
        leakage = np.einsum("bwr,wrxy->bxy", comm_power, self.c2s_matrix)

        # Expand target dimension, shape [B,W,M,R]
        leakage = np.repeat(leakage[:,:,None,:], self.num_targets, axis=2)
        return leakage.astype(np.float32)
    

    def compute_ibi_interference(self, X_s, X_c, bs_position):
        """
        Compute inter-BS interference.
        Parameters
            X_s : (np.ndarray, [B,W,M,R]) sensing allocation
            X_c : (np.ndarray, [B,W,S,U,R]) communication allocation
            bs_position : (np.ndarray, [B,3])
        Returns
            ibi_interf : (np.ndarray, [B,W,1,R])
        """
        p_each = np.asarray(self.pEach, dtype=np.float32)
        # Sensing transmit power [B,W,M,R] -> [B,W,R]
        p_sensing = np.sum(X_s * p_each[None, :, None, None], axis=2)
        # Communication transmit power [B,W,S,U,R] -> [B,W,R]
        p_comm = np.sum(X_c * p_each[None, :, None, None, None], axis=(2, 3))

        # Total transmitted power occupying the same (w,r)-[B,W,R]
        p_total = p_sensing + p_comm
        bs_position = np.asarray(bs_position, dtype=np.float32)
        B = bs_position.shape[0]

        # BS-BS distance
        delta = (bs_position[:, None, :] - bs_position[None, :, :])
        dist = np.linalg.norm(delta, axis=-1)
        dist = np.maximum(dist, 1.0)

        # Inter-BS propagation gain
        path_gain = (self.g_tx * self.g_rx * (self.wavelength / (4.0 * np.pi * dist)) ** 2)
        if self.shadowing > 0:
            shadow_db = self.rng.normal(0.0, self.shadowing, size=(B, B))
            path_gain *= (10.0 ** (shadow_db / 10.0))
        np.fill_diagonal(path_gain, 0.0)
        path_gain *= self.ibi_eta

        # IBI: I_bwr = Σ_j G_bj P_jwr
        ibi_interf = np.einsum("ij,jwr->iwr", path_gain, p_total)
        # [B,W,R] -> [B,W,1,R]
        ibi_interf = ibi_interf[:, :, None, :]

        return ibi_interf.astype(np.float32)

 
    def get_sensing_sinr(self, X_s, X_c, bs_pos, target_pos, pred_target_pos=None, comm_gain=None):
        """
        Compute sensing SINR.
        Parameters
            X_s : (np.ndarray, [B,W,M,R]) Sensing allocation matrix.
            X_c : (np.ndarray, [B,W,S,U,R]) Communication allocation matrix.
            bs_pos : (np.ndarray, [B,3])
            target_pos : (np.ndarray, [M,3])
            pred_target_pos : (np.ndarray | None, [M,3])
            comm_gain : (np.ndarray | None, [B,W,S,U,R]) Precomputed communication gain, reused for C2S computation.
        Returns
            effective_sensing_sinr : (np.ndarray, [B,M])
            sensing_sinr_prb : (np.ndarray, [B,W,M,R])    
        """
        sensing_gain = self.get_sensing_channel_gain(bs_pos, target_pos, pred_target_pos)
        noise_power_w = (self.noise_power * (2.0 ** np.asarray(self.bwp)))

        noise_power_w = (noise_power_w[None,:,None,None])

        p_prb = np.asarray(self.pEach, dtype=np.float32)[None, :, None, None]

        # Signal power, shape [B,W,M,R]
        signal_power = (X_s * sensing_gain * p_prb)

        # Communication leakage, shape [B,W,M,R]
        c2s_interf = self.compute_c2s_interference(X_c)

        # Inter-BS interference, shape [B,W,M,R]
        ibi_interf = self.compute_ibi_interference(X_s, X_c, bs_pos)
        total_interf = (noise_power_w + c2s_interf + ibi_interf)

        sensing_sinr_prb = np.where(X_s > 0, signal_power / np.maximum(total_interf, 1e-16), 0.0).astype(np.float32)
        
        # Γ_{b,m}, sum over W (BWP) and R (PRB)
        effective_sensing_sinr = np.sum(sensing_sinr_prb, axis=(1, 3)).astype(np.float32)

        return (effective_sensing_sinr, sensing_sinr_prb)
    
    def get_comm_sinr(self, X_c, X_s, bs_pos, ue_pos, pred_ue_pos=None):
            """
            Compute communication SINR.
            Parameters
                X_c : [B,W,S,U,R]
                X_s : [B,W,M,R]
                bs_pos : [B,3]
                ue_pos : [S,U,3]
            Returns
                effective_comm_sinr : [B,W,S,U]
                comm_sinr_prb : [B,W,S,U,R]
            """
            comm_gain = self.get_comm_channel_gain(bs_pos, ue_pos, pred_ue_pos)
            p_prb = np.asarray(self.pEach, dtype=np.float32)[None,:,None,None,None]

            noise_power_w = (self.noise_power * (2.0 ** np.asarray(self.bwp, dtype=np.float32)))
            noise_power_w = (noise_power_w[None,:,None,None,None])

            # Signal power
            signal_power = (X_c * comm_gain * p_prb)
            # IBI
            ibi_interf = self.compute_ibi_interference(X_s, X_c, bs_pos)
            # [B,W,1,R] -> [B,W,1,1,R]
            ibi_interf = (ibi_interf[:,:,None,:,:])

            total_interf = (noise_power_w + ibi_interf)

            comm_sinr_prb = np.where(X_c > 0, signal_power / np.maximum(total_interf, 1e-16),
                0.0).astype(np.float32)
            effective_comm_sinr = np.sum(comm_sinr_prb, axis=-1)

            return (effective_comm_sinr, comm_sinr_prb)
    
    
    def step(self, X_c, X_s, bs_pos, ue_pos, target_pos, pred_target_pos=None, pred_ue_pos=None):
        """
        Simulate physical layer channel.
        Parameters
            X_c : (np.ndarray, [B,W,S,U,R]) Communication allocation.
            X_s : (np.ndarray, [B,W,M,R]) Sensing allocation.
            bs_pos : (np.ndarray, [B,3])
            ue_pos : (np.ndarray, [S,U,3])
            target_pos : (np.ndarray, [M,3])
            pred_target_pos : (np.ndarray | None, [M,3])
            pred_ue_pos : (np.ndarray | None, [S,U,3])
        Returns
            dict comm_gain: [B,W,S,U,R]
                sensing_gain: [B,W,M,R]
                effective_sensing_sinr: [B,M]
                sensing_sinr_prb: [B,W,M,R]
        """

        # =====================================
        # Communication channel gain
        # =====================================

        comm_gain = self.get_comm_channel_gain(
            bs_pos,
            ue_pos,
            pred_ue_pos
        )

        # =====================================
        # Sensing channel gain
        # =====================================

        sensing_gain = self.get_sensing_channel_gain(
            bs_pos,
            target_pos,
            pred_target_pos
        )

        # =====================================
        # Sensing SINR
        # =====================================

        effective_sensing_sinr, sensing_sinr_prb = (
            self.get_sensing_sinr(
                X_s=X_s,
                X_c=X_c,
                bs_pos=bs_pos,
                target_pos=target_pos,
                pred_target_pos=pred_target_pos,
            )
        )

        # =====================================
        # Communication SINR
        # =====================================
        comm_sinr_prb = (self.get_comm_sinr(X_c, X_s, bs_pos, ue_pos, pred_ue_pos))
        return {
            "comm_gain": comm_gain,
            "sensing_gain": sensing_gain,
            "effective_sensing_sinr":
                effective_sensing_sinr,
            "sensing_sinr_prb":
                sensing_sinr_prb,
            "comm_sinr_prb":
                comm_sinr_prb
        }