import numpy as np


class CommunicationManager:
    def __init__(self, config: dict):
        """
        Initialize network system parameters and QoS requirements.
        Argument:
            (config) (dict): Configuration dictionary containing system parameters and requirements.
        Return:
            None
        """
        self.config = config or {}
        sys_config = config.get("system_parameters", config)
        req_config = sys_config["requirements"] 

        self.num_bs = sys_config["num_bs"]
        self.num_ues = sys_config["num_ue"]
        self.num_targets = sys_config["num_target"]
        self.n_prb = sys_config["num_prb"]

        self.B_prb = sys_config["prb_bandwidth_khz"] * 1e3
        self.R_min = req_config["min_ue_rate_mbps"] * 1e6

        self.noise_psd = float(sys_config["noise_psd_dbm_hz"])
        self.noise_fig = float(sys_config["noise_figure_db"])
        self.noise_power_ue = float(sys_config["noise_power_per_prb_watt"])

    def compute_sinr(self, X_c, P_c, comm_gain, inter_bs_interf=None):
        """
        Compute Communication SINR and Shannon spectral efficiency per PRB 
        Argument:
            (X_c) (numpy.ndarray)[B, K, R]: Binary PRB allocation matrix for communication (1 if allocated, 0 otherwise).
            (P_c) (numpy.ndarray)[B, K, R]: Transmit power allocation matrix for communication.
            (channel_gains) (numpy.ndarray)[B, K, R] or [B, K]: Effective channel gain between BS and UE.
            (inter_bs_interf) (numpy.ndarray or float)[B, K, R] or None: Inter-BS and radar-to-comm interference power in Watts.
        Return:
            (comm_sinr) (numpy.ndarray)[B, K, R]: Communication SINR
            (spectral_eff) (numpy.ndarray)[B, K, R]: Shannon spectral efficiency 
        Note:
            Only PRBs with X_c == 1 and P_c > 0 have non-zero SINR.
            Uses np.maximum on the denominator to avoid zero-division errors.
        """
        if inter_bs_interf is None:
            inter_bs_interf = 0.0
        signal_pow = P_c * comm_gain
        denomiator = inter_bs_interf + self.noise_power_ue
        comm_sinr = np.where((X_c == 1) & (P_c > 0), signal_pow / np.maximum(denomiator, 1e-16), 0.0).astype(np.float32)
        spectral_eff = np.log2(1.0 + np.maximum(comm_sinr, 0.0)).astype(np.float32)
        return comm_sinr, spectral_eff

    def compute_user_rates(self, X_c, spectral_eff):
        """
        Compute throughput (data rate) for each ue
        Argument:
            (X_c) (numpy.ndarray)[B, K, R]: Binary PRB allocation matrix for communication.
            (spectral_eff) (numpy.ndarray)[B, K, R]: Spectral efficiency per PRB 
        Return:
            (user_rates) (numpy.ndarray)[K]: Achievable data rate for each UE
        """
        rate_per_prb = X_c * self.B_prb * spectral_eff  # 3d tensor: (B, K, R)
        user_rates = np.sum(rate_per_prb, axis=(0, 2)).astype(np.float32)
        return user_rates

    def evaluate_qos(self, user_rates, X_c, flag=None):
        """
        Evaluate Quality of Service (QoS) satisfaction and compute penalty for DRL reward.
        Argument:
            (user_rates) (numpy.ndarray)[K]: Actual data rate for each UE in bps.
            (X_c) (numpy.ndarray)[B, K, R]: Binary PRB allocation matrix for communication.
            (flag) (numpy.ndarray)[K] or None: Serving indicator flag pi_k (1 if served, 0 otherwise).
        Return:
            (comm_penalty) (float): Normalized QoS violation penalty for DRL reward (in Mbps scale).
            (all_satisfied) (bool): True if all active UEs satisfy R_k >= R_min, False otherwise.
            (metrics) (dict): Summary dictionary containing penalty, satisfaction rate, sum rate, and rates in Mbps.
        """
        if flag is None:
            flag = (np.sum(X_c, axis=(0, 2)) > 0).astype(np.float32)

        require_rate = self.R_min * flag
        rate_gap = np.maximum(0.0, require_rate - user_rates)  # bps
        # change to Mbps
        comm_penalty = float(np.sum(rate_gap) / 1e6)

        served_idx = np.where(flag == 1)[0]
        if len(served_idx) > 0:
            satisfied_flags = user_rates[served_idx] >= require_rate[served_idx]
            all_satisfied = bool(np.all(satisfied_flags))
            satisfaction_rate = float(np.mean(satisfied_flags))
        else:
            all_satisfied = True
            satisfaction_rate = 1.0

        metrics = {
            "comm_penalty": comm_penalty,
            "all_satisfied": all_satisfied,
            "satisfaction_rate": satisfaction_rate,
            "sum_rate_mbps": float(np.sum(user_rates) / 1e6),
            "user_rates_mbps": user_rates / 1e6,
            "served_user": flag,
        }
        return comm_penalty, all_satisfied, metrics

    def evaluate(self, X_c, P_c, channels, inter_bs_interf=None):
        """
        Argument:
            (X_c) (numpy.ndarray)[B, K, R]: Binary PRB allocation matrix for communication.
            (P_c) (numpy.ndarray)[B, K, R]: Transmit power allocation matrix for communication
            (channels) (numpy.ndarray or dict)[B, K, R] or [B, K]: Channel gains or channel package from channel.py.
            (inter_bs_interf) (numpy.ndarray or float)[B, K, R] or None: External interference power 

        Return:
            (user_rates) (numpy.ndarray)[K]: Actual data rate of each UE in bps.
            (penalty) (float): QoS violation penalty for PPO reward.
            (satisfied) (bool): Whether all served UEs meet QoS threshold.
            (metrics) (dict):  communication metrics dictionary.
        """
        # 1. calculate sinr and spectral efficiency
        sinr, spectral_eff = self.compute_sinr(X_c, P_c, channels, inter_bs_interf)
        # 2. calculate user_rate (throughput) of each user
        user_rates = self.compute_user_rates(X_c, spectral_eff)
        # 3. calculate qos penalty
        penalty, satisfied, metrics = self.evaluate_qos(user_rates, X_c)
        return user_rates, penalty, satisfied, metrics