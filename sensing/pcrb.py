
import numpy as np


class PCRB:

    def __init__(
        self,
        state_dim,
        measurement_dim,
        num_target,
        dt,
        process_noise,
        initial_covariances,
        sepBS,
        bwp_ids=None
    ):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.num_target = num_target
        self.dt = dt
        self.sepBS = sepBS

        # =====================================================
        # BWP configuration
        # =====================================================

        self.bwp_ids = (
            list(bwp_ids)
            if bwp_ids is not None
            else ["BWP0"]
        )

        if len(self.bwp_ids) == 0:
            raise ValueError("At least one BWP is required")

        if len(set(self.bwp_ids)) != len(self.bwp_ids):
            raise ValueError("bwp_ids must be unique")

        self.num_bwp = len(self.bwp_ids)

        # =====================================================
        # Process noise covariance
        # =====================================================

        self.Q = np.array(
            process_noise,
            dtype=float,
            copy=True
        )

        if (
            self.Q.shape != (state_dim, state_dim)
            or not np.isfinite(self.Q).all()
            or not np.allclose(
                self.Q,
                self.Q.T,
                rtol=1e-10,
                atol=1e-15
            )
            or np.linalg.eigvalsh(self.Q).min() < -1e-12
        ):
            raise ValueError(
                "process_noise must be a finite symmetric "
                "positive-semidefinite state covariance"
            )

        # =====================================================
        # Initial covariance
        # =====================================================

        self.p_diag_init = np.diag(
            initial_covariances["p_diag_init"]
        )

        if self.p_diag_init.shape != (state_dim, state_dim):
            raise ValueError("Invalid initial covariance dimension")

        # =====================================================
        # State and covariance storage
        # =====================================================

        self.state_list = []

        self.covariance_list = [
            self.p_diag_init.copy()
            for _ in range(self.num_target)
        ]

        # =====================================================
        # Per-BWP Fisher information storage
        # =====================================================

        self.bwp_information = {
            bwp_id: [
                np.zeros(
                    (state_dim, state_dim),
                    dtype=float
                )
                for _ in range(num_target)
            ]
            for bwp_id in self.bwp_ids
        }

        # =====================================================
        # Position and velocity uncertainty
        # =====================================================

        self.u_pos = np.full(
            self.num_target,
            np.trace(self.p_diag_init[:3, :3]),
            dtype=float
        )

        self.u_vel = np.full(
            self.num_target,
            np.trace(self.p_diag_init[3:6, 3:6]),
            dtype=float
        )

        # =====================================================
        # Constant velocity model
        # State = [x, y, z, vx, vy, vz]
        # =====================================================

        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # =====================================================
        # Result history
        # =====================================================

        self.position_history = []
        self.velocity_history = []
        self.covariance_history = []

        self.bwp_information_history = []

    # =========================================================
    # Numerical utility
    # =========================================================

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    # =========================================================
    # Update target states
    # =========================================================

    def update_state(self, states_list):

        states = np.asarray(
            states_list,
            dtype=float
        )

        if states.shape != (
            self.num_target,
            self.state_dim
        ):
            raise ValueError(
                "states_list must have shape "
                "(num_target, state_dim)"
            )

        self.state_list = [
            state.copy()
            for state in states
        ]

        self.covariance_list = [
            self.p_diag_init.copy()
            for _ in range(self.num_target)
        ]

        self.u_pos = np.full(
            self.num_target,
            np.trace(self.p_diag_init[:3, :3]),
            dtype=float
        )

        self.u_vel = np.full(
            self.num_target,
            np.trace(self.p_diag_init[3:6, 3:6]),
            dtype=float
        )

        # Reset BWP information
        for bwp_id in self.bwp_ids:
            self.bwp_information[bwp_id] = [
                np.zeros(
                    (self.state_dim, self.state_dim),
                    dtype=float
                )
                for _ in range(self.num_target)
            ]

        # Reset history
        self.position_history = []
        self.velocity_history = []
        self.covariance_history = []
        self.bwp_information_history = []

    # =========================================================
    # Set linearization states
    # =========================================================

    def set_linearization_states(self, states_list):

        states = np.asarray(
            states_list,
            dtype=float
        )

        if states.shape != (
            self.num_target,
            self.state_dim
        ):
            raise ValueError(
                "states_list must have shape "
                "(num_target, state_dim)"
            )

        self.state_list = [
            state.copy()
            for state in states
        ]

    # =========================================================
    # Prediction
    # =========================================================

    def predict(self, target_id):

        if len(self.state_list) != self.num_target:
            raise RuntimeError(
                "Call update_state() before predict()."
            )

        if not 0 <= target_id < self.num_target:
            raise IndexError("Invalid target_id")

        state = self.state_list[target_id]
        covariance = self.covariance_list[target_id]

        predicted_state = self.F @ state

        predicted_covariance = (
            self.F @ covariance @ self.F.T
            + self.Q
        )

        predicted_covariance = self._symmetrize(
            predicted_covariance
        )

        self.state_list[target_id] = predicted_state

        self.covariance_list[target_id] = (
            predicted_covariance
        )

        self._update_uncertainty(target_id)

    # =========================================================
    # Ideal measurement function
    # Measurement = [range, azimuth, elevation]
    # =========================================================

    def ideal_measurement_function(
        self,
        state,
        bs_position
    ):

        x, y, z = state[:3]

        xb, yb, zb = np.asarray(
            bs_position,
            dtype=float
        )

        dx = x - xb
        dy = y - yb
        dz = z - zb

        horizontal_distance = np.sqrt(
            dx**2 + dy**2
        )

        range_ = np.sqrt(
            dx**2 + dy**2 + dz**2
        )

        azimuth = np.arctan2(dy, dx)

        elevation = np.arctan2(
            dz,
            horizontal_distance
        )

        return np.array([
            range_,
            azimuth,
            elevation
        ])

    # =========================================================
    # Measurement Jacobian
    # =========================================================

    def measurement_jacobian(
        self,
        state,
        bs_position
    ):

        x, y, z = state[:3]

        xb, yb, zb = np.asarray(
            bs_position,
            dtype=float
        )

        dx = x - xb
        dy = y - yb
        dz = z - zb

        rho_sq = dx**2 + dy**2
        rho = np.sqrt(rho_sq)

        r_sq = rho_sq + dz**2
        r = np.sqrt(r_sq)

        eps = 1e-10

        rho = max(rho, eps)
        rho_sq = max(rho_sq, eps)
        r_sq = max(r_sq, eps)
        r = max(r, eps)

        H = np.zeros(
            (self.measurement_dim, self.state_dim),
            dtype=float
        )

        # Range
        H[0, 0] = dx / r
        H[0, 1] = dy / r
        H[0, 2] = dz / r

        # Azimuth
        H[1, 0] = -dy / rho_sq
        H[1, 1] = dx / rho_sq

        # Elevation
        H[2, 0] = (
            -dx * dz / (r_sq * rho)
        )

        H[2, 1] = (
            -dy * dz / (r_sq * rho)
        )

        H[2, 2] = rho / r_sq

        return H

    # =========================================================
    # Measurement update with multiple BWPs
    #
    # Old format:
    # (bs_position, measurement, R)
    #
    # New format:
    # (bwp_id, bs_position, measurement, R)
    # =========================================================

    def update(self, target_id, measurements):

        if len(self.state_list) != self.num_target:
            raise RuntimeError(
                "Call update_state() before update()."
            )

        if not 0 <= target_id < self.num_target:
            raise IndexError("Invalid target_id")

        if measurements is None or len(measurements) == 0:
            self._update_uncertainty(target_id)
            return

        state = self.state_list[target_id]

        covariance = self.covariance_list[target_id]

        # Predicted information matrix
        information = np.linalg.pinv(covariance)

        # Reset information contribution of each BWP
        for bwp_id in self.bwp_ids:
            self.bwp_information[bwp_id][target_id] = (
                np.zeros(
                    (self.state_dim, self.state_dim),
                    dtype=float
                )
            )

        # Process measurements from different BWPs
        for item in measurements:

            if len(item) == 3:

                bs_position, measurement, R = item

                # Backward compatibility
                bwp_id = self.bwp_ids[0]

            elif len(item) == 4:

                bwp_id, bs_position, measurement, R = item

            else:
                raise ValueError(
                    "Measurement must be "
                    "(bs_position, measurement, R) or "
                    "(bwp_id, bs_position, measurement, R)"
                )

            if bwp_id not in self.bwp_ids:
                raise ValueError(
                    f"Unknown BWP ID: {bwp_id}"
                )

            R = np.asarray(
                R,
                dtype=float
            )

            if R.shape != (
                self.measurement_dim,
                self.measurement_dim
            ):
                raise ValueError(
                    "Each R must have shape "
                    "(measurement_dim, measurement_dim)"
                )

            if not np.all(np.isfinite(R)):
                raise ValueError(
                    "R contains non-finite values"
                )

            R = self._symmetrize(R)

            if np.linalg.eigvalsh(R).min() <= 0:
                raise ValueError(
                    "Measurement covariance R must "
                    "be positive definite"
                )

            # Measurement Jacobian
            H = self.measurement_jacobian(
                state,
                bs_position
            )

            # Fisher information from this measurement
            J_measurement = (
                H.T @ np.linalg.solve(R, H)
            )

            # Accumulate total information
            information += J_measurement

            # Accumulate per-BWP information
            self.bwp_information[bwp_id][target_id] += (
                J_measurement
            )

        information = self._symmetrize(information)

        # Posterior covariance bound
        updated_covariance = np.linalg.pinv(
            information
        )

        self.covariance_list[target_id] = (
            self._symmetrize(updated_covariance)
        )

        self._update_uncertainty(target_id)

    # =========================================================
    # Update uncertainty
    # =========================================================

    def _update_uncertainty(self, target_id):

        covariance = self.covariance_list[target_id]

        self.u_pos[target_id] = np.trace(
            covariance[:3, :3]
        )

        self.u_vel[target_id] = np.trace(
            covariance[3:6, 3:6]
        )

    # =========================================================
    # Get position and velocity uncertainty
    # =========================================================

    def get_uncer(self):

        return (
            self.u_pos.copy(),
            self.u_vel.copy()
        )

    # =========================================================
    # Get covariance bound
    # =========================================================

    def get_covariance_bound(self, target_id):

        return self.covariance_list[target_id].copy()

    # =========================================================
    # Get Fisher information from one BWP
    # =========================================================

    def get_bwp_information(
        self,
        target_id,
        bwp_id
    ):

        if bwp_id not in self.bwp_ids:
            raise ValueError(
                f"Unknown BWP ID: {bwp_id}"
            )

        return self.bwp_information[bwp_id][
            target_id
        ].copy()

    # =========================================================
    # Get trace of Fisher information from each BWP
    # =========================================================

    def get_bwp_information_trace(self, target_id):

        return {
            bwp_id: np.trace(
                self.bwp_information[bwp_id][target_id]
            )
            for bwp_id in self.bwp_ids
        }

    # =========================================================
    # Record results
    # =========================================================

    def record(self):

        self.position_history.append(
            self.u_pos.copy()
        )

        self.velocity_history.append(
            self.u_vel.copy()
        )

        self.covariance_history.append(
            np.asarray(
                self.covariance_list
            ).copy()
        )

        self.bwp_information_history.append({
            bwp_id: np.asarray(
                self.bwp_information[bwp_id]
            ).copy()
            for bwp_id in self.bwp_ids
        })

    # =========================================================
    # Get results
    # =========================================================

    def get_results(self):

        if not self.position_history:

            return {
                "position_uncertainty": np.empty(
                    (0, self.num_target)
                ),

                "velocity_uncertainty": np.empty(
                    (0, self.num_target)
                ),

                "covariance_bound": np.empty(
                    (
                        0,
                        self.num_target,
                        self.state_dim,
                        self.state_dim
                    )
                ),

                "bwp_information": {
                    bwp_id: np.empty(
                        (
                            0,
                            self.num_target,
                            self.state_dim,
                            self.state_dim
                        )
                    )
                    for bwp_id in self.bwp_ids
                }
            }

        return {
            "position_uncertainty": np.asarray(
                self.position_history
            ),

            "velocity_uncertainty": np.asarray(
                self.velocity_history
            ),

            "covariance_bound": np.asarray(
                self.covariance_history
            ),

            "bwp_information": {
                bwp_id: np.asarray([
                    frame[bwp_id]
                    for frame in self.bwp_information_history
                ])
                for bwp_id in self.bwp_ids
            }
        }
