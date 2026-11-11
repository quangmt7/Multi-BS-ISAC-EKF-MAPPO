from .env import BaseISACEnv
import numpy as np

class PPOISACEnv(BaseISACEnv):
    def _get_global_state(self):
        """
        Construct global state vector S_t for the Centralized Critic:
        S_t = [ {U~_m^pos}, {U~_m^vel}, {A~_m}, {d~_{b,m}^pred},
                {H~_{b,k}^avg}, {H~_{b,m}^avg}, {P~_b^res}, {N~_b^res} ]
                
        Total dimension: 3*M + 2*B*M + B*K + 2*B.
        
        Returns:
            global_state (np.ndarray): Shape [global_state_dim]
        """
        # 1. Uncertainty & AoI: [M], [M], [M]
        u_pos, u_vel = self._get_uncertainty_summary()
        u_pos_norm = np.clip(u_pos / self.u_pos_max, 0.0, 1.0).astype(np.float32)
        u_vel_norm = np.clip(u_vel / self.u_vel_max, 0.0, 1.0).astype(np.float32)
        aoi_norm = np.clip(self.aoi / self.a_max, 0.0, 1.0).astype(np.float32)

        # 2. Predicted relative distances d~_{b,m}^pred: [B * M]
        pred_target_pos = self.pred_target_states[:, :3]
        d_pred = np.zeros((self.num_bs, self.num_targets), dtype=np.float32)
        for b in range(self.num_bs):
            for m in range(self.num_targets):
                dist = float(np.linalg.norm(pred_target_pos[m] - self.bs_positions[b]))
                d_pred[b, m] = np.clip(dist / self.d_max, 0.0, 1.0)
        d_pred_flat = d_pred.flatten()

        # 3. Spectral efficiency features normalized to [0, 1]: [B * K] and [B * M]
        h_comm_norm = np.clip(self.last_h_comm / self.se_comm_scale, 0.0, 1.0).astype(np.float32)
        h_sens_norm = np.clip(self.last_h_sens / self.se_sens_scale, 0.0, 1.0).astype(np.float32)
        h_comm_flat = h_comm_norm.flatten()
        h_sens_flat = h_sens_norm.flatten()

        # 4. Residual resources: [B] and [B]
        p_res_norm = np.clip(self.p_res / self.p_max, 0.0, 1.0).astype(np.float32)
        n_res_norm = np.clip(self.n_res / self.total_prb, 0.0, 1.0).astype(np.float32)

        global_state = np.concatenate([
            u_pos_norm,     # [M]
            u_vel_norm,     # [M]
            aoi_norm,       # [M]
            d_pred_flat,    # [B * M]
            h_comm_flat,    # [B * K]
            h_sens_flat,    # [B * M]
            p_res_norm,     # [B]
            n_res_norm      # [B]
        ]).astype(np.float32)

        return global_state
