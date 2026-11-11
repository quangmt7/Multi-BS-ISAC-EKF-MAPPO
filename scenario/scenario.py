
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.special import erfinv, erfc
import math


SPEED_OF_LIGHT_M_S = 3.0e8       # Vận tốc ánh sáng c (m/s)
BOLTZMANN_K        = 1.38e-23     # Hằng số Boltzmann k_B (J/K)
TEMPERATURE_K      = 290.0        # Nhiệt độ chuẩn T_0 (Kelvin)


# =============================================================================
# Lớp BaseStation: Đại diện cho một trạm phát cơ sở BS
# =============================================================================
class BaseStation:
    """
    Đại diện cho một trạm cơ sở (Base Station - BS) trong hệ thống ISAC.

    Parameters
    ----------
    bs_id : int
        Chỉ số định danh của BS (0-indexed: b in {0, ..., B-1}).
    position : np.ndarray, shape (3,)
        Tọa độ 3D cố định: [x_b, y_b, z_b] (mét).
    max_tx_power_w : float
        Công suất phát tối đa của BS: P_b^{max} (Watt).
    """
    def __init__(self, bs_id: int, position: np.ndarray, max_tx_power_w: float = 2.0):
        self.bs_id          = int(bs_id)
        self.position       = np.asarray(position, dtype=np.float64)
        self.max_tx_power_w = float(max_tx_power_w)

    def __repr__(self) -> str:
        return f"BaseStation(id={self.bs_id}, pos={self.position.tolist()}, P_max={self.max_tx_power_w:.2f}W)"


# =============================================================================
# Lớp UserEquipment: Đại diện cho người dùng truyền thông UE
# =============================================================================
class UserEquipment:
    """
    Communication UE.
    Each UE belongs to exactly one slice.
    """
    def __init__(self, ue_id: int, slice_id: int, service_type: str, position: np.ndarray):
        self.ue_id = int(ue_id)
        self.slice_id = int(slice_id)
        self.service_type = str(service_type)
        self.position = np.asarray(position, dtype=np.float64)


# =============================================================================
# Lớp SensingTarget: Đại diện cho mục tiêu cảm biến di động
# =============================================================================
class SensingTarget:
    """
    Đại diện cho một mục tiêu cần cảm biến/theo dõi (Sensing Target).

    - Trạng thái 6 chiều: s_m = [x, y, z, v_x, v_y, v_z]^T (mét, m/s).
    - Radar Cross Section (RCS): sigma_rcs (m^2).
    - Age of Information (AoI): A_m(t). Reset về 0 khi sensing thành công

    Parameters
    ----------
    target_id : int
        Chỉ số định danh mục tiêu (0-indexed: m in {0, ..., M-1}).
    state : np.ndarray, shape (6,)
        Vector trạng thái 6 chiều: [x, y, z, vx, vy, vz].
    rcs : float
        Diện tích phản xạ hiệu dụng radar (m^2).
    init_aoi : int
        AoI khởi tạo ban đầu (mặc định = 0).
    """
    def __init__(self, target_id: int, state: np.ndarray, rcs: float = 1.0, init_aoi: int = 0):
        self.target_id = int(target_id)
        self.state     = np.asarray(state, dtype=np.float64)
        self.rcs       = float(rcs)
        self.aoi       = int(init_aoi)

    @property
    def position(self) -> np.ndarray:
        """Tọa độ 3D hiện tại: [x, y, z] (mét)."""
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        """Vận tốc 3D hiện tại: [vx, vy, vz] (m/s)."""
        return self.state[3:]

    def __repr__(self) -> str:
        pos_str = np.round(self.position, 1).tolist()
        vel_str = np.round(self.velocity, 2).tolist()
        return f"SensingTarget(id={self.target_id}, pos={pos_str}, vel={vel_str}, aoi={self.aoi})"


# =============================================================================
# Lớp Scenario: Kịch bản chính quản lý toàn bộ hệ thống Multi-BS ISAC
# =============================================================================
class Scenario:
    """
    Kịch bản mô phỏng Multi-BS ISAC với EKF Tracking và MAPPO Resource Allocation.
    """
    def __init__(
        self,
        cfg_system: dict,
        cfg_ekf: Optional[dict] = None,
        cfg_mappo: Optional[dict] = None,
        seed: int = 42
    ):
        self.rng = np.random.default_rng(seed)
        sp = cfg_system.get("system_parameters", cfg_system)

        # 1. Quy mô hệ thống 
        self.B = int(sp.get("num_bs", 5))

        self.num_embb_slices = int(sp["num_embb"])
        self.num_urllc_slices = int(sp["num_urllc"])
        self.num_ue_1embb = int(sp["num_ue_1embb"])
        self.num_ue_1urllc = int(sp["num_ue_1urllc"])
        self.num_slices = (self.num_embb_slices + self.num_urllc_slices)
        self.num_ues_max = max(self.num_ue_1embb, self.num_ue_1urllc)
        self.K = (self.num_embb_slices * self.num_ue_1embb + self.num_urllc_slices * self.num_ue_1urllc)
        self.M = int(sp["num_target"])
        

        self.slice_type = np.concatenate([np.zeros(self.num_embb_slices, dtype=np.int32),
                          np.ones(self.num_urllc_slices, dtype=np.int32)])
        self.embb_slice_idx = np.where(self.slice_type == 0)[0]
        self.urllc_slice_idx = np.where(self.slice_type == 1)[0]
        # Số PRB của MỖI BS
        self.nPrb = list(sp.get("nPrb"))
        self.max_prb = max(self.nPrb)
        self.bw = list(sp.get("bwp"))

        self.bw_bs_hz = float(sp.get("bandwidth_mhz_bs", 20.0)) * 1e6

        self.prb_bw_hz = float(sp.get("prbBW", 180)) * 1e3
        # 2. Cấu hình vô tuyến và OFDM 5G NR
        self.bw_hz     = [self.prb_bw_hz * 2**w for w in self.bw]
        self.fc_hz     = float(sp.get("carrier_frequency_ghz", 3.5)) * 1e9
        self.dt        = float(sp["slot_duration_ms"]) * 1e-3
        self.lambda_m  = SPEED_OF_LIGHT_M_S / self.fc_hz

        # Config pEach uses mW; internal physical calculations use W.
        self.pEachBWP = np.asarray(sp["pEach"], dtype=float) * 1e-3
        self.p_max_w = float(np.dot(self.pEachBWP, self.nPrb))

        # Công suất tạp âm nhiệt trên 1 PRB: sigma^2 = N_0 * NF * B_PRB 
        if "noise_power_per_prb_watt" in sp:
            self.sigma2_w = float(sp["noise_power_per_prb_watt"])
        else:
            n0_dbm_hz = float(sp.get("noise_psd_dbm_hz", -174.0))
            nf_db     = float(sp.get("noise_figure_db", 7.0))
            sigma2_dbm = n0_dbm_hz + nf_db + 10.0 * np.log10(self.prb_bw_hz)
            self.sigma2_w = 10.0 ** (sigma2_dbm / 10.0) * 1e-3

        # 3. Tham số kênh truyền, Beamforming và Can nhiễu
        ci = sp.get("channel_and_interference", {})
        self.pl_model         = str(ci.get("pathloss_model", "3gpp_macro"))
        self.pl_const_db      = float(ci.get("pathloss_const_db", 128.1))           # Hằng số 128.1 dB 
        self.pl_exp_comm      = float(ci.get("pathloss_exponent_comm", 37.6))        # Hệ số 37.6 * log10(d_km) 
        self.pl_exp_sensing   = float(ci.get("pathloss_exponent_sensing", 2.0))     # alpha_s = 2.0 (suy hao 2 chiều ~ d^4)
        self.shadowing_std_db = float(ci.get("shadowing_std_db", 6.0))              # sigma_SF = 6.0 dB 
        self.rcs_sqm          = float(ci.get("target_rcs_sqm", 1.0))

        # Đặc tính ăng-ten & Beamforming 
        self.num_antennas_bs  = int(ci.get("num_antennas_bs", 64))                  # N_b = 64
        # G_ant: Độ lợi của 1 phần tử antenna đơn lẻ 
        self.g_ant_lin        = float(ci.get("antenna_element_gain_linear", 10.0 ** (float(ci.get("antenna_element_gain_dbi", 0.0)) / 10.0)))
        # G_b / G_max: Độ lợi phát cực đại của mảng antenna BS
        self.ant_gain_tx_lin  = float(ci.get("antenna_gain_tx_linear", 10.0 ** (float(ci.get("antenna_gain_tx_dbi", 15.0)) / 10.0)))
        self.ant_gain_rx_lin  = float(ci.get("antenna_gain_rx_linear", 10.0 ** (float(ci.get("antenna_gain_rx_dbi", 15.0)) / 10.0)))
        self.beam_pattern_q   = float(ci.get("beam_pattern_q", 2.0))                # q = 2.0 
        
        # Độ lợi búp sóng phát cực đại G_max = G_b (15.0 dBi = 31.62)
      
        if "antenna_gain_tx_linear" in ci or "antenna_gain_tx_dbi" in ci:
            self.g_max_lin = float(self.ant_gain_tx_lin) # G_max = G_b = 15.0 dBi (tuyến tính 31.62)
        else:
            self.g_max_lin = float(self.num_antennas_bs * self.g_ant_lin)

        # Tham số can nhiễu C2S, S2C và IBI
        self.eta0_c2s         = float(ci.get("c2s_leakage_eta0", 1.0e-3))           # eta_0 = 10^-3 (-30 dB)
        self.alpha_c2s        = float(ci.get("c2s_leakage_alpha", 1.0))             # alpha = 1.0
        self.s2c_leakage      = float(ci.get("s2c_leakage_factor", 1.0e-2))         # Can nhiễu Radar-to-Comm 
        self.eta_ibi          = float(ci.get("ibi_coupling_eta", 1.0e-2))           # eta_IBI = 10^-2 (-20 dB)

        # Tham số UE (ue_parameters)
        ue_cfg = sp.get("ue_parameters", {})
        self.ue_ant_gain_lin = float(ue_cfg.get("antenna_gain_rx_linear", 1.0))      # G_rx_ue = 0 dBi = 1.0
        self.ue_min_dist_bs  = float(ue_cfg.get("min_distance_to_bs_m", 10.0))

        # Kịch bản không gian 3D
        sc = sp.get("scenario", {})
        self.area_x    = float(sc.get("area_size_x_m", 1000.0))
        self.area_y    = float(sc.get("area_size_y_m", 1000.0))
        self.area_z    = float(sc.get("area_size_z_m", 100.0))
        self.d_max     = float(sc.get("diagonal_max_m", math.sqrt(self.area_x**2 + self.area_y**2 + self.area_z**2)))
        self.bs_h      = float(sc.get("bs_height_m", 25.0))
        self.ue_h      = float(sc.get("ue_height_m", 1.5))
        self.tgt_h_min = float(sc.get("target_height_min_m", 10.0))
        self.tgt_h_max = float(sc.get("target_height_max_m", 60.0))
        self.tgt_v_min = float(sc.get("target_velocity_min_mps", 5.0))
        self.tgt_v_max = float(sc.get("target_velocity_max_mps", 25.0))

        # Ràng buộc chỉ tiêu hệ thống (requirements)
        req = sp.get("requirements", {})
        self.req_r_min_bps      = float(req.get("min_ue_rate_mbps", 1.0)) * 1e6      # R_k^min = 1.0 Mbps
        self.req_p_m_min        = float(req.get("min_sensing_success_prob", 0.90))  # P_m^min = 0.90
        self.req_prob_false_alm = float(req.get("max_prob_false_alarm", 1.0e-4))    # P_F = 1.0e-4

        # Các hằng số chuẩn hóa & trọng số thưởng/phạt mặc định 
        self.alpha     = 0.4
        self.beta      = 0.4
        self.gamma     = 0.2
        self.c1        = 0.2
        self.c2        = 0.2
        self.c3        = 0.1
        self.c4        = 0.1
        self.a_max     = 20.0
        self.u_pos_max = 20.0
        self.u_vel_max = 10.0

        
        if cfg_mappo is not None:
            mp = cfg_mappo.get("mappo_parameters", cfg_mappo)
            rw = mp.get("reward_weights", {})
            self.alpha = float(rw.get("alpha_aoi", self.alpha))
            self.beta  = float(rw.get("beta_uncertainty", self.beta))
            self.gamma = float(rw.get("gamma_power", self.gamma))

            pen = mp.get("penalties", {})
            self.c1 = float(pen.get("c1_rate_qos", pen.get("penalty_rate_qos", self.c1)))
            self.c2 = float(pen.get("c2_sensing_reliability", pen.get("penalty_sensing_reliability", self.c2)))
            self.c3 = float(pen.get("c3_pos_uncertainty", pen.get("penalty_uncertainty", self.c3)))
            self.c4 = float(pen.get("c4_vel_uncertainty", pen.get("penalty_uncertainty", self.c4)))

            norm_cfg = mp.get("normalization", {})
            self.a_max     = float(norm_cfg.get("a_max", self.a_max))
            self.u_pos_max = float(norm_cfg.get("u_pos_max", self.u_pos_max))
            self.u_vel_max = float(norm_cfg.get("u_vel_max", self.u_vel_max))
            self.d_max     = float(norm_cfg.get("diagonal_max_m", self.d_max))

        # 4. Mô hình động học CV (Constant Velocity) cho EKF
        dt = self.dt
        self.F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ])

        q_s = 0.4
        if cfg_ekf is not None:
            ekf_p = cfg_ekf.get("ekf_parameters", cfg_ekf)
            q_s = float(ekf_p.get("process_noise", {}).get("continuous_noise_spectral_density", 0.4))
        self.q_s = q_s
        self.Q_m = q_s * np.block([
            [(dt**3 / 3.0) * np.eye(3), (dt**2 / 2.0) * np.eye(3)],
            [(dt**2 / 2.0) * np.eye(3),  dt * np.eye(3)]
        ])

        # Bảng tra khoảng cách PRB cho nhiễu C2S leakage 
        r_idx = np.arange(self.max_prb)

        dist_rr = np.abs(r_idx[:, None] - r_idx[None, :])
        self.c2s_lookup = self.eta0_c2s * np.exp(-self.alpha_c2s * dist_rr)

        # 5. Khởi tạo các thực thể mạng
        self.base_stations = self._init_base_stations()
        self.ues           = self._init_ues()
        self.targets       = self._init_targets()

        # 6. Kênh truyền thông:
        # channel_comm: ma trận hệ số fading nhanh h^c_{b,k,r} ~ CN(0, 1)
        # channel_gain_comm: độ lợi kênh hiệu dụng G^c_{b,k,r}(t) 
        self.channel_comm = np.zeros((self.B, len(self.nPrb), self.num_slices, 
                                      self.num_ues_max, self.max_prb), dtype=np.complex64)
        self.channel_gain_comm = np.zeros((self.B, len(self.nPrb), self.num_slices, self.num_ues_max,
                                        self.max_prb), dtype=np.float32)
        self.sinr_comm = np.zeros((self.B, len(self.nPrb), self.num_slices, self.num_ues_max,
                                    self.max_prb), dtype=np.float32)
        self._update_channel_comm()

        # Bộ đệm kết quả slot hiện tại
        self.sinr_sensing = np.zeros((self.B, len(self.nPrb), self.M, self.max_prb), dtype=np.float32)
        self.prob_detect  = np.zeros((self.B, self.M), dtype=np.float64)
        self.prob_success = np.zeros(self.M, dtype=np.float64)
        self.current_slot = 0

    # =========================================================================
    # Khởi tạo vị trí các thực thể (BS, UE, Target)
    # =========================================================================
    def _init_base_stations(self) -> List[BaseStation]:
        """
        Khởi tạo B trạm BS. Với B=5: 1 trạm trung tâm và 4 trạm tại 4 góc.
        """
        bs_list = []
        if self.B == 5:
            positions = [
                [self.area_x * 0.5, self.area_y * 0.5, self.bs_h],  
                [self.area_x * 0.2, self.area_y * 0.2, self.bs_h], 
                [self.area_x * 0.8, self.area_y * 0.2, self.bs_h],  
                [self.area_x * 0.2, self.area_y * 0.8, self.bs_h],  
                [self.area_x * 0.8, self.area_y * 0.8, self.bs_h],  
            ]
        else:
            r_ring = min(self.area_x, self.area_y) * 0.35
            angles = np.linspace(0, 2 * np.pi, self.B, endpoint=False)
            positions = [
                [
                    self.area_x * 0.5 + r_ring * np.cos(a),
                    self.area_y * 0.5 + r_ring * np.sin(a),
                    self.bs_h
                ]
                for a in angles
            ]

        for b_idx, pos in enumerate(positions):
            bs_list.append(BaseStation(
                bs_id=b_idx,
                position=np.array(pos, dtype=np.float64),
                max_tx_power_w=self.p_max_w
            ))
        return bs_list

    def _init_ues(self):
        """
        Returns
        list[list[UserEquipment]] [S][U] 
        """
        ue_matrix = []
        bs_pos_2d = np.array([bs.position[:2] for bs in self.base_stations])
        ue_id = 0
        for s in range(self.num_slices):
            num_ue_per_slice = (self.num_ue_1embb if self.slice_type[s] == 0 else self.num_ue_1urllc)
            service_type = ("embb" if self.slice_type[s] == 0 else "urllc")
            slice_users = []
            for u in range(num_ue_per_slice):
                for _ in range(500):
                    x = self.rng.uniform(0.0, self.area_x)
                    y = self.rng.uniform(0.0, self.area_y)
                    dists = np.linalg.norm(bs_pos_2d - np.array([x, y]), axis=1)
                    if np.all(dists >= self.ue_min_dist_bs):
                        break
                slice_users.append(UserEquipment(ue_id=ue_id, slice_id=s, service_type=service_type,
                        position=np.array([x, y, self.ue_h], dtype=np.float64)))
                ue_id += 1
            ue_matrix.append(slice_users)

        return ue_matrix

    def _init_targets(self) -> List[SensingTarget]:
        """
        Khởi tạo M mục tiêu cảm biến với vị trí và vận tốc ban đầu ngẫu nhiên.
        """
        target_list = []
        for m in range(self.M):
            x_m   = self.rng.uniform(0.0, self.area_x)
            y_m   = self.rng.uniform(0.0, self.area_y)
            z_m   = self.rng.uniform(self.tgt_h_min, self.tgt_h_max)

            v_mag = self.rng.uniform(self.tgt_v_min, self.tgt_v_max)
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            vx_m  = v_mag * np.cos(theta)
            vy_m  = v_mag * np.sin(theta)
            vz_m  = self.rng.uniform(-0.5, 0.5)

            state_m = np.array([x_m, y_m, z_m, vx_m, vy_m, vz_m], dtype=np.float64)
            target_list.append(SensingTarget(
                target_id=m,
                state=state_m,
                rcs=self.rcs_sqm,
                init_aoi=0
            ))
        return target_list

    # =========================================================================
    # Kênh truyền thông & Kênh Radar
    # =========================================================================
    def _update_channel_comm(self):
        """
        channel_comm: [B,W,S,U,Rmax]
        channel_gain_comm: [B,W,S,U,Rmax]
        """
        self.channel_comm.fill(0.0)
        self.channel_gain_comm.fill(0.0)
        for b, bs in enumerate(self.base_stations):
            for s in range(self.num_slices):
                num_ue = (self.num_ue_1embb if self.slice_type[s] == 0 else self.num_ue_1urllc)
                for u in range(num_ue):
                    ue = self.ues[s][u]
                    d_m = max(
                        np.linalg.norm(bs.position - ue.position),
                        self.ue_min_dist_bs
                    )
                    d_km = d_m / 1000.0
                    pl_db = (self.pl_const_db + self.pl_exp_comm * np.log10(d_km))
                    g_path = 10.0 ** (-pl_db / 10.0)
                    if self.shadowing_std_db > 0:
                        chi_db = self.rng.normal(0.0, self.shadowing_std_db)
                        sh_lin = 10.0 ** (chi_db / 10.0)
                    else:
                        sh_lin = 1.0
                    for w, n_prb in enumerate(self.nPrb):
                        h = (self.rng.standard_normal(n_prb) + 1j * self.rng.standard_normal(n_prb)
                            ) / np.sqrt(2.0)
                        self.channel_comm[b, w, s, u, :n_prb] = h
                        h2 = np.abs(h) ** 2
                        self.channel_gain_comm[b, w, s, u, :n_prb] = (g_path * sh_lin * h2
                            * self.g_max_lin * self.ue_ant_gain_lin)

    def compute_beamforming_gain(
        self,
        bs_pos: np.ndarray,
        true_pos: np.ndarray,
        pred_pos: Optional[np.ndarray] = None
    ) -> float:

        if pred_pos is None:
            return float(self.g_max_lin)

        diff_pred = pred_pos - bs_pos
        norm_pred = np.linalg.norm(diff_pred)
        diff_true = true_pos - bs_pos
        norm_true = np.linalg.norm(diff_true)

        if norm_pred < 1e-6 or norm_true < 1e-6:
            return float(self.g_max_lin)

        u_beam = diff_pred / norm_pred
        u_true = diff_true / norm_true

        cos_psi = np.clip(np.dot(u_beam, u_true), -1.0, 1.0)
        gain = self.g_max_lin * (np.abs(cos_psi) ** self.beam_pattern_q)
        return float(gain)

    def compute_prob_detection(
        self,
        gamma_s_bm: np.ndarray,
        prob_false_alarm: Optional[float] = None
    ) -> np.ndarray:
        """
        Tính xác suất phát hiện Neyman-Pearson:
        P^D_{b,m}(t) = 0.5 * erfc[ erfc^{-1}(2 * P_F) - sqrt(Gamma^s_{b,m}(t)) ]
        """
        if prob_false_alarm is None:
            prob_false_alarm = self.req_prob_false_alm

        erfc_inv_2pf = np.sqrt(2.0) * erfinv(1.0 - 2.0 * prob_false_alarm)
        gamma_clipped = np.clip(gamma_s_bm, 0.0, None)
        arg = erfc_inv_2pf - np.sqrt(gamma_clipped)
        prob_detect = 0.5 * erfc(arg)

        self.prob_detect = np.clip(prob_detect, 0.0, 1.0)
        return self.prob_detect

    def compute_prob_success(self, prob_detect: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Tính xác suất sensing thành công tổng hợp từ các BS:
        P^{suc}_m(t) = 1 - prod_{b in B} (1 - P^D_{b,m}(t))
        """
        if prob_detect is None:
            prob_detect = self.prob_detect

        prob_miss_all = np.prod(1.0 - prob_detect, axis=0)  
        self.prob_success = np.clip(1.0 - prob_miss_all, 0.0, 1.0)
        return self.prob_success

    # =========================================================================
    # Cập nhật AoI 
    # =========================================================================
    def update_aoi(
        self,
        prob_success: Optional[np.ndarray] = None,
        min_success_prob: Optional[float] = None,
        reset_value: int = 0
    ) -> np.ndarray:
        """
        Cập nhật Age of Information A_m(t):
            A_m(t+1) = 0           khi D_m(t) = 1 (P^{suc}_m >= P^{min}_m)
            A_m(t+1) = A_m(t) + 1  khi D_m(t) = 0
        """
        if prob_success is None:
            prob_success = self.prob_success
        if min_success_prob is None:
            min_success_prob = self.req_p_m_min

        sensing_success = (prob_success >= min_success_prob)

        aoi_vec = np.array([tgt.aoi for tgt in self.targets], dtype=np.float64)
        aoi_new = np.where(sensing_success, float(reset_value), aoi_vec + 1.0)

        for m_idx, tgt in enumerate(self.targets):
            tgt.aoi = int(aoi_new[m_idx])

        return aoi_new

    def compute_comm_rate(self, sinr_comm: np.ndarray, X_c: np.ndarray) -> np.ndarray:
        """
        Compute communication throughput.
        Parameters
            sinr_comm : (np.ndarray, [B,W,S,U_max,R_max])
            X_c : (np.ndarray, [B,W,S,U_max,R_max])
        Returns
            rate_bps : (np.ndarray, [S,U_max])      
        """
        rate_bps = np.zeros((self.num_slices, self.num_ues_max), dtype=np.float32)
        for w, n_prb in enumerate(self.nPrb):
            bw_prb = self.bw_hz[w]
            se = np.log2(1.0 + np.maximum(sinr_comm[:, w, :, :, :n_prb], 0.0))
            rate_bps += np.sum( X_c[:, w, :, :, :n_prb] * bw_prb * se, axis=(0, 3))
        rate_bps *= self.get_ue_mask()
        return rate_bps

    # =========================================================================
# Động học mục tiêu và Vector đo đạc EKF
# =========================================================================
    def step_targets(self) -> None:
        """Propagate every target with independent process noise.

        Argument:
            (None).

        Return:
            (None).

        Note:
            Each target reflects its velocity at the simulation boundaries.
        """
        for tgt in self.targets:
            w_m = self.rng.multivariate_normal(
                np.zeros(6), self.Q_m
            )
            new_state = self.F @ tgt.state + w_m

            if new_state[0] < 0.0 or new_state[0] > self.area_x:
                new_state[3] = -new_state[3]
                new_state[0] = np.clip(new_state[0], 0.0, self.area_x)

            if new_state[1] < 0.0 or new_state[1] > self.area_y:
                new_state[4] = -new_state[4]
                new_state[1] = np.clip(new_state[1], 0.0, self.area_y)

            if new_state[2] < self.tgt_h_min or new_state[2] > self.tgt_h_max:
                new_state[5] = -new_state[5]
                new_state[2] = np.clip(
                    new_state[2], self.tgt_h_min, self.tgt_h_max
                )

            tgt.state = new_state

    def get_measurement_vectors(self) -> np.ndarray:
        """
        Tính vector đo đạc lý tưởng e_{b,m} = [r, theta, phi]^T :
            r_{b,m}     = sqrt(dx^2 + dy^2 + dz^2)
            theta_{b,m} = arctan2(dy, dx)
            phi_{b,m}   = arctan2(dz, sqrt(dx^2 + dy^2))

        Returns
        -------
        meas : np.ndarray, shape (B, M, 3)
        """
        bs_pos  = self.get_bs_positions()     # (B, 3)
        tgt_pos = self.get_target_positions() # (M, 3)

        meas = np.zeros((self.B, self.M, 3), dtype=np.float64)
        for b_idx in range(self.B):
            for m_idx in range(self.M):
                dx = tgt_pos[m_idx, 0] - bs_pos[b_idx, 0]
                dy = tgt_pos[m_idx, 1] - bs_pos[b_idx, 1]
                dz = tgt_pos[m_idx, 2] - bs_pos[b_idx, 2]

                r = np.sqrt(dx**2 + dy**2 + dz**2)
                r = max(r, 1e-3)

                theta = np.arctan2(dy, dx)
                d_xy  = np.sqrt(dx**2 + dy**2)
                phi   = np.arctan2(dz, d_xy)

                meas[b_idx, m_idx, :] = [r, theta, phi]

        return meas

    # =========================================================================
    # State Vector toàn cục chuẩn hóa S_t 
    # =========================================================================
    def compute_state_vector(
        self,
        u_pos: np.ndarray,                             
        u_vel: np.ndarray,                             
        pred_target_positions: np.ndarray,            
        power_alloc: np.ndarray,                      
        alloc_matrix: np.ndarray,                      
        u_pos_max: Optional[float] = None,
        u_vel_max: Optional[float] = None,
        a_max: Optional[float] = None,
    ) -> np.ndarray:
        
        M = self.M
        B = self.B
        S = self.num_slices
        U = self.num_ues_max
        u_pos_max = u_pos_max or self.u_pos_max
        u_vel_max = u_vel_max or self.u_vel_max
        a_max = a_max or self.a_max

        # 1. Tỉ lệ về độ bất định vị trí & vận tốc: U~_m^pos, U~_m^vel in [0, 1]
        u_pos_norm = np.clip(u_pos / u_pos_max, 0.0, 1.0)
        u_vel_norm = np.clip(u_vel / u_vel_max, 0.0, 1.0)

        # 2. Tỉ lệ về tuổi thông tin: A~_m(t) in [0, 1]
        aoi_vec = self.get_aoi()
        aoi_norm = np.clip(aoi_vec / a_max, 0.0, 1.0)

        # 3. Chuẩn hóa khoảng cách tương đối dự đoán: d~_{b,m}^pred in [0, 1]
        # d~_{b,m}^pred(t) = ||p^_m(t|t-1) - p_b||_2 / d_max
        bs_pos = self.get_bs_positions()  # (B, 3)
        d_pred = np.linalg.norm(pred_target_positions[None, :, :] - bs_pos[:, None, :], axis=2)  # (B, M)
        d_pred_norm = np.clip(d_pred / self.d_max, 0.0, 1.0).flatten()  # (25,)

        # 4. Hiệu suất phổ H~_{b,k}^avg(t) = log2(1 + SINR_bar_{b,k})
        h_avg_comm = np.zeros((B, S, U), dtype=np.float32)
        for b in range(B):
            for s in range(S):
                num_ue = (self.num_ue_1embb if self.slice_type[s] == 0 else self.num_ue_1urllc)
                for u in range(num_ue):
                    sinr_bsu = self.sinr_comm[b, :, s, u, :]
                    valid = sinr_bsu > 0
                    if np.any(valid):
                        sinr_mean = np.mean(sinr_bsu[valid])
                    else:
                        sinr_mean = 0.0
                    h_avg_comm[b, s, u] = np.log2(1.0 + sinr_mean)
        h_avg_comm_flat = h_avg_comm.flatten()

        # 5. Hiệu suất phổ cảm biến H~_{b,m}^avg(t) = log2(1 + SINR_bar_{b,m}) 
        h_avg_sens = np.zeros((B, M), dtype=np.float64)
        for b in range(B):
            for m in range(M):
                sinr_mean = 0.0
                mask_bm = (alloc_matrix[b] == (m + 1))
                if np.any(mask_bm):
                    sinr_bm = self.sinr_sensing[b, :, m, :]
                    valid = sinr_bm > 0
                    if np.any(valid):
                        sinr_mean = np.mean(sinr_bm[valid])
                h_avg_sens[b, m] = np.log2(1.0 + sinr_mean)
        h_avg_sens_flat = h_avg_sens.flatten()  

        # 6. Tỉ lệ công suất còn lại: P~_b^res(t) = P_b^res / P_b^max
        p_used_b = np.sum(power_alloc, axis=1)  # (B,)
        p_res_b = np.clip((self.p_max_w - p_used_b) / self.p_max_w, 0.0, 1.0)  # (5,)

        # 7. Tỉ lệ PRB còn lại: N~_b^res(t) = N_b^res / N_PRB
        n_used_b = np.sum(alloc_matrix != 0, axis=1)  # (B,)
        total_prb = np.sum(self.nPrb)
        n_res_b = np.clip((total_prb - n_used_b) / float(total_prb), 0.0, 1.0)

        s_t = np.concatenate([
            u_pos_norm.flatten(),
            u_vel_norm.flatten(),
            aoi_norm.flatten(),
            d_pred_norm,
            h_avg_comm_flat,
            h_avg_sens_flat,
            p_res_b.flatten(),
            n_res_b.flatten()
        ]).astype(np.float32)
        return s_t

    # =========================================================================
    # Hàm mục tiêu và Hàm phần thưởng Reward 
    # =========================================================================
    def compute_reward(
        self,
        rate_bps: np.ndarray,             # shape: (S,U_max)
        prob_success: np.ndarray,         # shape: (M,)
        u_pos: np.ndarray,                # shape: (M,)
        u_vel: np.ndarray,                # shape: (M,)
        power_alloc: np.ndarray,          
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        c3: Optional[float] = None,
        c4: Optional[float] = None,
        a_max: Optional[float] = None,
        u_pos_max: Optional[float] = None,
        u_vel_max: Optional[float] = None,
        p_max: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
       
        alpha = self.alpha if alpha is None else alpha
        beta  = self.beta  if beta  is None else beta
        gamma = self.gamma if gamma is None else gamma
        c1    = self.c1    if c1    is None else c1
        c2    = self.c2    if c2    is None else c2
        c3    = self.c3    if c3    is None else c3
        c4    = self.c4    if c4    is None else c4
        a_max = a_max or self.a_max
        u_pos_max = u_pos_max or self.u_pos_max
        u_vel_max = u_vel_max or self.u_vel_max
        p_max = p_max or self.p_max_w

        # AoI trung bình
        aoi_vec = self.get_aoi()
        a_bar = float(np.mean(aoi_vec))

        # Độ bất định trung bình
        u_bar_pos = float(np.mean(u_pos))
        u_bar_vel = float(np.mean(u_vel))

        # Tổng công suất tiêu thụ trung bình mỗi BS
        p_total_used = float(np.sum(power_alloc))
        p_used_per_bs = p_total_used / float(self.B)

        # 1. Thành phần mục tiêu chuẩn hóa R_obj 
        term_aoi = alpha * (a_bar / a_max)
        term_unc = beta * ((u_bar_pos / u_pos_max) + (u_bar_vel / u_vel_max))
        term_pow = gamma * (p_used_per_bs / p_max)
        r_obj = - (term_aoi + term_unc + term_pow)

        # 2. Thành phần phạt Hinge Loss R_penalty 
        # c1: Phạt vi phạm QoS truyền thông: R_k < R_min
        r_min = self.req_r_min_bps
        viol_rate = np.maximum(0.0, (r_min - rate_bps) / r_min)
        pen_rate = c1 * float(np.sum(viol_rate))

        # c2: Phạt suy giảm xác suất sensing: P_suc < P_min
        p_min_sens = self.req_p_m_min
        viol_sens = np.maximum(0.0, (p_min_sens - prob_success) / p_min_sens)
        pen_sens = c2 * float(np.sum(viol_sens))

        # c3: Phạt vượt ngưỡng bất định vị trí: U_pos > U_pos_max
        viol_upos = np.maximum(0.0, (u_pos - u_pos_max) / u_pos_max)
        pen_upos = c3 * float(np.sum(viol_upos))

        # c4: Phạt vượt ngưỡng bất định vận tốc: U_vel > U_vel_max
        viol_uvel = np.maximum(0.0, (u_vel - u_vel_max) / u_vel_max)
        pen_uvel = c4 * float(np.sum(viol_uvel))

        r_penalty = pen_rate + pen_sens + pen_upos + pen_uvel
        r_t = r_obj - r_penalty

        info = {
            "reward_total": r_t,
            "r_obj": r_obj,
            "r_penalty": r_penalty,
            "pen_rate_qos": pen_rate,
            "pen_sensing_rel": pen_sens,
            "pen_pos_unc": pen_upos,
            "pen_vel_unc": pen_uvel,
            "a_bar": a_bar,
            "u_bar_pos": u_bar_pos,
            "u_bar_vel": u_bar_vel,
            "p_used_total": p_total_used
        }
        return r_t, info

    # =========================================================================
    # Step & Reset Interface cho Environment
    # =========================================================================
    def step(
        self, rate_bps: np.ndarray,  sinr_sensing: np.ndarray,
        sinr_sensing_agg: np.ndarray, prob_false_alarm=None,
        min_success_prob=None) -> Dict:
        """
        Thực hiện một time slot đầy đủ của kịch bản ISAC.
        """
        self.sinr_comm.fill(0.0)
        self.sinr_sensing.fill(0.0)
        self.prob_detect.fill(0.0)
        self.prob_success.fill(0.0)
        # Cập nhật kênh truyền thông hiệu dụng với 3GPP Macro & Log-normal Shadowing
        self._update_channel_comm()

        # Xác suất phát hiện và xác suất sensing thành công 
        prob_detect  = self.compute_prob_detection(sinr_sensing_agg, prob_false_alarm)
        prob_success = self.compute_prob_success(prob_detect)

        # Cập nhật Age of Information 
        aoi_new = self.update_aoi(prob_success, min_success_prob, reset_value=0)

        # Cập nhật động học mục tiêu sang slot t+1
        self.step_targets()
        self.current_slot += 1

        return {
            "slot"             : self.current_slot,
            "sinr_sensing"     : sinr_sensing,
            "sinr_sensing_agg" : sinr_sensing_agg,
            "prob_detect"      : prob_detect,
            "prob_success"     : prob_success,
            "rate_bps"         : rate_bps,
            "aoi"              : aoi_new,
            "target_states"    : self.get_target_states(),
            "bs_positions"     : self.get_bs_positions(),
            "ue_positions"     : self.get_ue_positions(),
        }

    def reset(self, seed: Optional[int] = None) -> Dict:
        """
        Reset kịch bản cho episode mới.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.ues          = self._init_ues()
        self.targets      = self._init_targets()
        self._update_channel_comm()
        self.current_slot = 0
        self.sinr_comm.fill(0.0)
        self.sinr_sensing.fill(0.0)
        self.prob_detect.fill(0.0)
        self.prob_success.fill(0.0)

        return {
            "slot"          : 0,
            "target_states" : self.get_target_states(),
            "bs_positions"  : self.get_bs_positions(),
            "ue_positions"  : self.get_ue_positions(),
            "aoi"           : self.get_aoi(),
        }

    # =========================================================================
    # Helper getters
    # =========================================================================
    def get_bs_positions(self) -> np.ndarray:
        return np.stack([bs.position for bs in self.base_stations])

    def get_ue_positions(self) -> np.ndarray:
        ue_pos = np.zeros((self.num_slices, self.num_ues_max, 3), dtype=np.float32)
        for s, users in enumerate(self.ues):
            for u, ue in enumerate(users):
                ue_pos[s, u] = ue.position

        return ue_pos
    
    def get_ue_mask(self):
        ue_mask = np.zeros((self.num_slices, self.num_ues_max), dtype=np.float32)
        for s, users in enumerate(self.ues):
            ue_mask[s, :len(users)] = 1.0

        return ue_mask

    def get_target_states(self) -> np.ndarray:
        return np.stack([tgt.state for tgt in self.targets])

    def get_target_positions(self) -> np.ndarray:
        return np.stack([tgt.position for tgt in self.targets])

    def get_aoi(self) -> np.ndarray:
        return np.array([tgt.aoi for tgt in self.targets], dtype=np.float64)

    def get_distance_bs_target(self) -> np.ndarray:
        bs_pos  = self.get_bs_positions()
        tgt_pos = self.get_target_positions()
        return np.linalg.norm(bs_pos[:, None, :] - tgt_pos[None, :, :], axis=2)

    def get_distance_bs_ue(self) -> np.ndarray:
        """
        Returns
        distance : (np.ndarray, [B,S,U_max])   
        """
        bs_pos = self.get_bs_positions()
        ue_pos = self.get_ue_positions()
        distance = np.linalg.norm(bs_pos[:, None, None, :] - ue_pos[None, :, :, :], axis=-1)

        return distance.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"Scenario("
            f"B={self.B}, "
            f"S={self.num_slices}, "
            f"K={self.K}, "
            f"M={self.M}, "
            f"W={len(self.nPrb)}, "
            f"Rmax={self.max_prb}, "
            f"slot={self.current_slot})"
        )
