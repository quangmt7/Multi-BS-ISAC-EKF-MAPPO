from .env import BaseISACEnv
import numpy as np

class HierISACEnv(BaseISACEnv):
    def reset(self, seed=None):
        result = super().reset(seed)
        self._frame_steps = None
        self._uses_frames = False
        self.last_frame_summary = None
        return result

    def begin_frame(self, ratios, num_subframes):
        """Normalize SAC ratios per BS, floor PRB quotas; hold caps for this frame.

        ratios: [B,W,S+M], slices before targets. The trainer supplies the
        frame length explicitly; no assumed physical duration or extra reset.
        """
        if not self._has_reset:
            raise RuntimeError("reset before begin_frame")
        if self._frame_steps is not None:
            raise RuntimeError("finish the active frame before replacing its quota")
        if isinstance(num_subframes, bool) or not isinstance(num_subframes, (int, np.integer)) or num_subframes <= 0:
            raise ValueError("num_subframes must be a positive integer")
        ratios = np.asarray(ratios, dtype=float)
        if ratios.shape != self.BWP_slice.shape or not np.isfinite(ratios).all() or np.any(ratios < 0):
            raise ValueError("SAC ratios must be finite non-negative [B,W,S+M]")
        if not self.bwp_mode and np.any(np.count_nonzero(ratios, axis=1) > 1):
            raise ValueError("single-BWP SAC ratios must select at most one BWP per service")
        # Scale first to avoid overflow when summing large valid ratios.
        scale = ratios.max(axis=(1, 2), keepdims=True)
        scaled = np.divide(ratios, scale, out=np.zeros_like(ratios), where=scale > 0)
        total = scaled.sum(axis=(1, 2), keepdims=True)
        self.frame_ratios = np.divide(scaled, total, out=np.zeros_like(ratios), where=total > 0)
        quota = np.floor(self.frame_ratios * np.asarray(self.nPrb)[None, :, None]).astype(np.int64)
        self.set_budget(quota)
        self._frame_length = int(num_subframes)
        self._uses_frames = True
        self._frame_steps = 0
        self._frame_cost = np.zeros(4)
        self._frame_aoi = np.zeros(self.num_targets)
        self._frame_u_pos = np.zeros(self.num_targets)
        self._frame_u_vel = np.zeros(self.num_targets)
        self._frame_service = np.zeros(self.num_slices)
        self.last_frame_summary = None
        return quota.copy()

    def step(self, actions_prb):
        if getattr(self, "_uses_frames", False) and self._frame_steps is None:
            raise RuntimeError("begin_frame is required before the next hierarchical step")
        result = super().step(actions_prb)
        if self._frame_steps is None:
            return result
        self._frame_steps += 1
        self._frame_cost += self.cost_by_bs.sum(axis=0)
        self._frame_aoi += self.aoi
        up, uv = self._get_uncertainty_summary()
        self._frame_u_pos += up
        self._frame_u_vel += uv
        counts = self.ue_mask.sum(axis=1)
        self._frame_service += self.served_ues.sum(axis=1) / counts
        info = result[-1]
        info["frame_complete"] = False
        if self._frame_steps == self._frame_length or result[3] or result[4]:
            info["frame_summary"] = self.end_frame()
            info["frame_complete"] = True
        return result

    def end_frame(self):
        """Finish a complete or partial frame using actual sampled subframe count."""
        if self._frame_steps is None or self._frame_steps == 0:
            raise RuntimeError("no sampled frame to finish")
        n = self._frame_steps
        cost = self._frame_cost / n
        aoi, up, uv = self._frame_aoi / n, self._frame_u_pos / n, self._frame_u_vel / n
        delta = self._frame_service / n
        normalized_cost = np.divide(cost, self.cost_max, out=np.zeros(4), where=self.cost_max > 0)
        reward, terms = self._reward_components(aoi, up, uv, normalized_cost, delta, frame=True)
        summary = dict(subframes=n, duration_seconds=n*self.slot_duration,
            cost_mean=cost.copy(), normalized_cost=normalized_cost.copy(),
            delta_E=delta[:self.num_embb].copy(), delta_U=delta[self.num_embb:].copy(),
            service_ratios=np.concatenate((delta[:self.num_embb], delta[self.num_embb:])),
            aoi_mean=aoi.copy(), u_pos_mean=up.copy(), u_vel_mean=uv.copy(),
            reward=reward, **terms)
        self.last_frame_summary = summary
        self._frame_steps = None
        self.set_budget(None)
        return summary

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
