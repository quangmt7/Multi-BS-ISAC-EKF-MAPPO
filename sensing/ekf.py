import numpy as np
from comUtils.common import wrap_angle
import math

class EKF:
    """
    This class is used to calculate uncertainty and update target's state
    Argument : 
        state_dim : number of state's dimension
        measurement_dim : number of measurement's demension
        num_target : number of target
        dt : time frame
        process_noise : all noise matrix, include all noise for six components
        initial_states : initial states for all target
        initial_covariances : initial covariances for all target
        sepBS : (bool) indicates if this EKF is used in common or separate BS
            True -> separate BS, False -> common 
    """
    def __init__(self, state_dim, measurement_dim, num_target, dt, process_noise, 
                 initial_covariances, sepBS):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.num_target = num_target
        self.dt = dt
        self.con_noise_spec_dense = process_noise["continuous_noise_spectral_density"]
        self.q_diag = np.diag(process_noise["q_diag"])
        self.p_diag_init = np.diag(initial_covariances["p_diag_init"])

        # Initial state estimate for each target, also state storage
        self.state_list = []
        # Each target has its own covariance matrix
        self.covariance_list = [
            self.p_diag_init.copy()
            for _ in range(self.num_target)
        ]

        self.u_pos = np.full(self.num_target, np.trace(self.p_diag_init[:3, :3]))
        self.u_vel = np.full(self.num_target, np.trace(self.p_diag_init[3:, 3:]))

        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

    def update_state(self, states_list):
        """
        Update states_list
        Parameter:
            states_list : (float)[self.num_target][6] State list
        """
        self.state_list = [
            np.asarray(state, dtype=float).copy()
            for state in states_list
        ]

    def predict(self, target_id):
        """
        Perform EKF prediction for one target.
        Argument:
            target_id (int) : id of the target
        """
        state = self.state_list[target_id]
        covariance = self.covariance_list[target_id]
        predicted_state = self.F @ state
        predicted_covariance = (self.F @ covariance @ self.F.T + self.q_diag)

        self.state_list[target_id] = predicted_state
        self.covariance_list[target_id] = predicted_covariance
        self.u_pos[target_id] = np.trace(predicted_covariance[:3, :3])
        self.u_vel[target_id] = np.trace(predicted_covariance[3:, 3:])

    def ideal_measurement_function(self, state, bs_position):
        """
        Map target state to ideal range, azimuth and elevation.
        Argument:
            state (float)[6] : target state
            bs_position (float)[3] : BS position
        Return:
            e (float)[3] : ideal measurement
        """
        x, y, z = state[:3]
        xb, yb, zb = bs_position
        dx = x - xb
        dy = y - yb
        dz = z - zb

        horizontal_distance = np.sqrt(dx**2 + dy**2)
        range_ = np.sqrt(dx**2 + dy**2 + dz**2)
        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(dz, horizontal_distance)

        return np.array([range_, azimuth, elevation])
    
    def measurement_jacobian(self, state, bs_position):
        """
        Calculate measurement Jacobian H.
        Argument : 
            state (float)[6] : target state
            bs_position (float)[3] : BS position
        Return: 
            H (float)[3][3] : Jacobian matrix
        """
        x, y, z = state[:3]
        xb, yb, zb = bs_position
        dx = x - xb
        dy = y - yb
        dz = z - zb

        rho_sq = dx**2 + dy**2
        rho = np.sqrt(rho_sq)
        r_sq = rho_sq + dz**2
        r = np.sqrt(r_sq)

        # Avoid division by zero
        eps = 1e-10

        rho = max(rho, eps)
        rho_sq = max(rho_sq, eps)
        r = max(r, eps)
        r_sq = max(r_sq, eps)

        H = np.zeros((3, 6))

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
        """
        Update one target using measurements from multiple BSs.
        Argument :
            target_id (int) : target id
            measurements: (float)[...][3] : all measurements for target id target_id
        Note:
        If there is no measurement -> no update
        If there is at least one measurement -> update
        """
        if not measurements:
            return

        state = self.state_list[target_id]
        covariance = self.covariance_list[target_id]

        measurement_list = []
        predicted_measurement_list = []
        jacobian_list = []
        covariance_list = []

        for bs_position, measurement, R in measurements:
            measurement = np.asarray(measurement, dtype=float)
            R = np.asarray(R, dtype=float)
            predicted_measurement = (self.ideal_measurement_function(state, bs_position))

            H = self.measurement_jacobian(state, bs_position)

            measurement_list.append(measurement)
            predicted_measurement_list.append(predicted_measurement)
            jacobian_list.append(H)
            covariance_list.append(R)

        # Batch measurement
        measurement_batch = np.concatenate(measurement_list)

        predicted_measurement_batch = np.concatenate(predicted_measurement_list)

        # Batch Jacobian
        H_batch = np.vstack(jacobian_list)

        # Block diagonal R
        num_measurements = len(measurements)

        R_batch = np.zeros((self.measurement_dim * num_measurements, 
                            self.measurement_dim * num_measurements))

        for i, R in enumerate(covariance_list):
            start = i * self.measurement_dim
            end = start + self.measurement_dim

            R_batch[start:end, start:end] = R

        # Innovation
        innovation = measurement_batch - predicted_measurement_batch
        innovation[1::self.measurement_dim] = wrap_angle(innovation[1::self.measurement_dim])
        innovation[2::self.measurement_dim] = wrap_angle(innovation[2::self.measurement_dim])

        # Innovation covariance
        S = (H_batch @ covariance @ H_batch.T + R_batch)

        # Kalman gain
        K = (covariance @ H_batch.T @ np.linalg.inv(S))

        # State update
        updated_state = (state + K @ innovation)

        # Covariance update
        I = np.eye(6)

        residual_operator = I - K @ H_batch
        updated_covariance = (residual_operator @ covariance @ residual_operator.T
                              + K @ R_batch @ K.T)

        self.state_list[target_id] = updated_state
        self.covariance_list[target_id] = updated_covariance
        self.u_pos[target_id] = updated_covariance[0][0] + updated_covariance[1][1] + \
                                updated_covariance[2][2]
        self.u_vel[target_id] = updated_covariance[3][3] + updated_covariance[4][4] + \
                                updated_covariance[5][5]

    def get_uncer(self):
        """
        Get position and velocity uncertainties
        Return 
            u_pos : (float)[num_target] position uncertainties
            u_vel : (float)[num_target] velocity uncertainties
        """
        return self.u_pos, self.u_vel

    def cal_rmse(self, true_pos):
        """
        Calculate the RMSE of each state's position
        Parameter
            true_pos : (float)[num_target, 3] a targets' true position
        Return
            rmse : RMSE of all targets
        """
        rmse = 0
        for i in range(len(self.num_target)):
            rmse_tar = 0
            for k in range(len(self.state_list[i])):
                rmse_tar += (self.state_list[i][k] - true_pos[i][k])
            rmse+=rmse_tar
        rmse = math.sqrt(rmse / self.num_target)
        return rmse
            


