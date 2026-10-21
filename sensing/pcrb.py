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
        sepBS
    ):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.num_target = num_target
        self.dt = dt
        self.sepBS = sepBS

        # Match the process-noise configuration used by EKF.
        self.con_noise_spec_dense = process_noise[
            "continuous_noise_spectral_density"
        ]
        self.q_diag = np.diag(process_noise["q_diag"])

        # Initial covariance for every target.
        self.p_diag_init = np.diag(
            initial_covariances["p_diag_init"]
        )

        # State storage and covariance-bound storage.
        self.state_list = []

        self.covariance_list = [
            self.p_diag_init.copy()
            for _ in range(self.num_target)
        ]

        # Position and velocity uncertainty.
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

        # Constant-velocity state transition matrix.
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Results recorded once per frame.
        self.position_history = []
        self.velocity_history = []
        self.covariance_history = []

    @staticmethod
    def _symmetrize(matrix):
        """Remove small numerical asymmetry."""
        return 0.5 * (matrix + matrix.T)

    def update_state(self, states_list):
        
        states = np.asarray(states_list, dtype=float)

        if states.shape != (self.num_target, self.state_dim):
            raise ValueError(
                "states_list must have shape "
                "(num_target, state_dim)"
            )

        self.state_list = [
            state.copy() for state in states
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

        self.position_history = []
        self.velocity_history = []
        self.covariance_history = []

    def set_linearization_states(self, states_list):
    
        states = np.asarray(states_list, dtype=float)

        if states.shape != (self.num_target, self.state_dim):
            raise ValueError(
                "states_list must have shape "
                "(num_target, state_dim)"
            )

        self.state_list = [
            state.copy() for state in states
        ]

    def predict(self, target_id):
        
        if len(self.state_list) != self.num_target:
            raise RuntimeError(
                "Call update_state() before predict()."
            )

        state = self.state_list[target_id]
        covariance = self.covariance_list[target_id]

        predicted_state = self.F @ state

        predicted_covariance = (
            self.F @ covariance @ self.F.T
            + self.q_diag
        )

        predicted_covariance = self._symmetrize(
            predicted_covariance
        )

        self.state_list[target_id] = predicted_state
        self.covariance_list[target_id] = predicted_covariance

        self._update_uncertainty(target_id)

    def ideal_measurement_function(self, state, bs_position):
        
        x, y, z = state[:3]
        xb, yb, zb = np.asarray(bs_position, dtype=float)

        dx = x - xb
        dy = y - yb
        dz = z - zb

        horizontal_distance = np.sqrt(dx**2 + dy**2)

        range_ = np.sqrt(
            dx**2 + dy**2 + dz**2
        )

        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(
            dz, horizontal_distance
        )

        return np.array([
            range_,
            azimuth,
            elevation
        ])

    def measurement_jacobian(self, state, bs_position):
        
        x, y, z = state[:3]
        xb, yb, zb = np.asarray(bs_position, dtype=float)

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
        H[2, 0] = -dx * dz / (r_sq * rho)
        H[2, 1] = -dy * dz / (r_sq * rho)
        H[2, 2] = rho / r_sq

        return H

    def update(self, target_id, measurements):
        if measurements is None or len(measurements) == 0:
            self._update_uncertainty(target_id)
            return

        if len(self.state_list) != self.num_target:
            raise RuntimeError(
                "Call update_state() before update()."
            )

        state = self.state_list[target_id]
        covariance = self.covariance_list[target_id]

        # Convert predicted covariance bound to information form.
        information = np.linalg.pinv(covariance)

        for bs_position, measurement, R in measurements:
            R = np.asarray(R, dtype=float)

            if R.shape != (
                self.measurement_dim,
                self.measurement_dim
            ):
                raise ValueError(
                    "Each R must have shape "
                    "(measurement_dim, measurement_dim)"
                )

            if not np.all(np.isfinite(R)):
                raise ValueError("R contains non-finite values")

            R = self._symmetrize(R)

            H = self.measurement_jacobian(
                state, bs_position
            )

            # Measurement information: H.T @ inv(R) @ H
            information += H.T @ np.linalg.solve(R, H)

        information = self._symmetrize(information)

        # Posterior covariance bound.
        updated_covariance = np.linalg.pinv(information)

        self.covariance_list[target_id] = self._symmetrize(
            updated_covariance
        )

        self._update_uncertainty(target_id)

    def _update_uncertainty(self, target_id):
        """Update position and velocity uncertainty for one target."""
        covariance = self.covariance_list[target_id]

        self.u_pos[target_id] = np.trace(
            covariance[:3, :3]
        )

        self.u_vel[target_id] = np.trace(
            covariance[3:6, 3:6]
        )

    def get_uncer(self):
        
        return self.u_pos.copy(), self.u_vel.copy()

    def get_covariance_bound(self, target_id):
        """Return a copy of the 6x6 covariance bound."""
        return self.covariance_list[target_id].copy()

    def record(self):
        
        self.position_history.append(self.u_pos.copy())
        self.velocity_history.append(self.u_vel.copy())

        self.covariance_history.append(
            np.asarray(self.covariance_list).copy()
        )

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
                    (0, self.num_target,
                     self.state_dim, self.state_dim)
                )
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
            )
        }
