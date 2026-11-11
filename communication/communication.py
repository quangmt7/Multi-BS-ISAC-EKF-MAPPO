import numpy as np


class CommunicationManager:
    def __init__(self, config: dict, bs_id: int):
        """
        Initialize network system parameters and QoS requirements.
        Argument:
            (config) (dict): Configuration dictionary containing system parameters and requirements.
            bs_id: (int) :  Base Station index (0 to num_bs - 1).
        Return:
            None
        """
        self.bs_id = bs_id
        self.config = config or {}
        sys_config = config.get("system_parameters", config)
        req_config = sys_config["requirements"] 

        self.pEach = np.asarray(sys_config["pEach"], dtype=float) * 1e-3  # mW -> W
        self.prbBW = sys_config["prbBW"]
        self.bwp = sys_config["bwp"]
        self.bwp_bandwidth = [self.prbBW * 1e3 * 2**w for w in self.bwp]  # Hz
        self.slot_duration = sys_config["slot_duration_ms"] / 1000.0

        self.R_min = req_config["min_ue_rate_mbps"]
        self.D_max = req_config["max_ue_delay"]
 
        self.noise_psd = float(sys_config["noise_psd_dbm_hz"])
        self.noise_fig = float(sys_config["noise_figure_db"])
        self.noise_power_ue = (float(sys_config["noise_power_per_prb_watt"])
                               * 2.0 ** np.asarray(self.bwp))

    def compute_sinr(self, X_c, comm_gain, inter_bs_interf=None):
        """
        Compute communication SINR and spectral efficiency.
        Parameters
            X_c : (np.ndarray, [B,W,S,U,R]) Communication allocation.
            comm_gain : (np.ndarray, [B,W,S,U,R]) Effective communication channel gain.
            inter_bs_interf : (np.ndarray | None, [B,W,S,U,R]) Inter-BS interference. Default is zero.
        Returns
            comm_sinr : (np.ndarray, [B,W,S,U,R]) Communication SINR.
            spectral_eff : (np.ndarray, [B,W,S,U,R]) Shannon spectral efficiency.      
        """
        if inter_bs_interf is None:
            inter_bs_interf = 0.0

        # Power of one PRB in each BWP, shape [1,W,1,1,1]
        p_prb = np.asarray(self.pEach, dtype=np.float32)[None,:,None,None,None]

        # Signal power p_bw * G_c
        signal_power = (X_c * comm_gain * p_prb)
        denominator = (self.noise_power_ue[None, :, None, None, None] + inter_bs_interf)
        comm_sinr = np.where(X_c > 0, signal_power / np.maximum(denominator, 1e-16), 0.0).astype(np.float32)
        spectral_eff = np.log2(1.0 + np.maximum(comm_sinr, 0.0)).astype(np.float32)

        return comm_sinr, spectral_eff

    def compute_embb_throughput(self, X_embb, spectral_eff):
        """
        Compute eMBB throughput.
        Parameters
            X_embb : (np.ndarray, [B,W,S_E,U,R]) eMBB allocation matrix.
            spectral_eff : (np.ndarray, [B,W,S_E,U,R]) Spectral efficiency.
        Returns
            embb_rate : (np.ndarray, [S_E,U]) Throughput in bps.
        """
        bwp_bw = np.asarray(self.bwp_bandwidth, dtype=np.float32)[None,:,None,None,None]

        rate_prb = (X_embb * bwp_bw * spectral_eff)

        embb_rate = np.sum(rate_prb, axis=(0, 1, 4))

        return embb_rate.astype(np.float32)
    
    def compute_urllc_bits(self, X_urllc, spectral_eff, slot_duration):
        """
        Compute received URLLC bits.
        Parameters
            X_urllc : (np.ndarray, [B,W,S_U,U,R]) URLLC allocation matrix.
            spectral_eff : (np.ndarray, [B,W,S_U,U,R]) Spectral efficiency.
            slot_duration : (float) Slot duration in seconds.
        Returns
            urllc_bits : (np.ndarray, [S_U,U]) Received bits during current slot.
        """
        bwp_bw = np.asarray(self.bwp_bandwidth, dtype=np.float32)[None,:,None,None,None]
        bits_prb = (X_urllc * bwp_bw * spectral_eff * slot_duration)
        urllc_bits = np.sum(bits_prb, axis=(0, 1, 4))

        return urllc_bits.astype(np.float32)


    def evaluate_qos(self, embb_rate, urllc_latency):
        """
        Evaluate eMBB and URLLC QoS.
        Parameters
            embb_rate : (np.ndarray, [S_E,U]) eMBB throughput in bps.
            urllc_latency : (np.ndarray, [S_U,U]) Current URLLC latency in ms.
        Returns
            qos_penalty : (float) Total QoS violation penalty.
            all_satisfied : (bool) True if all QoS requirements are met.
            metrics : (dict) QoS statistics.
        """
        embb_rate_mbps = np.asarray(embb_rate) / 1e6
        embb_gap = np.maximum(0.0, self.R_min - embb_rate_mbps)

        embb_penalty = (np.sum(embb_gap))
        urllc_gap = np.maximum(0.0, urllc_latency - self.D_max)

        urllc_penalty = np.sum(urllc_gap)

        qos_penalty = (embb_penalty + urllc_penalty)

        embb_satisfied = (embb_rate_mbps >= self.R_min)

        urllc_satisfied = (urllc_latency <= self.D_max)

        all_satisfied = bool(np.all(embb_satisfied) and np.all(urllc_satisfied))

        metrics = {
            "qos_penalty": float(qos_penalty),
            "embb_penalty": float(embb_penalty),
            "urllc_penalty": float(urllc_penalty),
            "embb_satisfaction": float(np.mean(embb_satisfied)),
            "urllc_satisfaction": float(np.mean(urllc_satisfied)),
            "mean_embb_rate_mbps": float(np.mean(embb_rate_mbps)),
            "mean_urllc_latency_ms": float(np.mean(urllc_latency)),
        }
        return (qos_penalty, all_satisfied, metrics)
    

    def evaluate(self, X_embb, X_urllc, comm_gain, urllc_latency, inter_bs_interf=None):
        """
        Evaluate communication performance.
        Parameters
            X_embb : (np.ndarray, [B,W,S_E,U,R]) 
            X_urllc : (np.ndarray, [B,W,S_U,U,R])
            comm_gain : (np.ndarray, [B,W,S,U,R])
            urllc_latency : (np.ndarray, [S_U,U])  
            inter_bs_interf : (np.ndarray | None, [B,W,S,U,R]) 
        Returns
            results : (dict)
        """
        X_comm = np.concatenate([X_embb, X_urllc], axis=2)
        sinr, spectral_eff = (self.compute_sinr(X_comm, comm_gain, inter_bs_interf))

        n_embb = X_embb.shape[2]

        embb_se = spectral_eff[:, :, :n_embb]
        urllc_se = spectral_eff[:, :, n_embb:]

        embb_rate = (self.compute_embb_throughput(X_embb, embb_se))

        urllc_bits = (self.compute_urllc_bits(X_urllc, urllc_se, self.slot_duration)
        )

        qos_penalty, satisfied, qos_metrics = (self.evaluate_qos(
                embb_rate,
                urllc_latency))

        metrics = {**qos_metrics, "embb_rate": embb_rate, "urllc_bits": urllc_bits}

        return (embb_rate, urllc_bits, qos_penalty, satisfied, metrics)
