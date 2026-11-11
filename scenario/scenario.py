
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

        self.bw_bs_hz = float(sp.get("bandwidth_mhz_bs", 20.0))

        self.prb_bw_hz = float(sp.get("prbBW", 180))
        # 2. Cấu hình vô tuyến và OFDM 5G NR
        self.bw_hz     = [self.prb_bw_hz * 2**w for w in self.bw]
        self.fc_hz     = float(sp.get("carrier_frequency_ghz", 3.5))
        self.dt        = float(sp.get("slot_duration_ms", 10.0)) * 1e-3
        self.lambda_m  = SPEED_OF_LIGHT_M_S / self.fc_hz

        # Công suất mỗi PRB của BWP của mỗi BS (mW)
        self.pEachBWP = list(sp["pEach"])
        self.p_max_w = self.p_max_w = float(np.dot(self.pEachBWP, self.nPrb))

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
        self.shadowing_std_db = float(ci.get("shadowing_std_db", 6.0))              # sigma_SF = 6.0 dB 
        self.rcs_sqm          = float(ci.get("target_rcs_sqm", 1.0))

        # Đặc tính ăng-ten & Beamforming 
        self.num_antennas_bs  = int(ci.get("num_antennas_bs", 64))                  # N_b = 64
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

        # 5. Khởi tạo các thực thể mạng
        self.base_stations = self._init_base_stations()
        self.ues           = self._init_ues()
        self.targets       = self._init_targets()

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

    def reset(self, seed: Optional[int] = None) -> Dict:
        """
        Reset kịch bản cho episode mới.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.ues          = self._init_ues()
        self.targets      = self._init_targets()
        self.current_slot = 0

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
    
    def get_process_noise_cov(self):
        return self.Q_m.copy()

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
