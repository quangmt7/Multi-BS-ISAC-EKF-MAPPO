import numpy as np
import scipy

class EKF:
    """
    This class is used to calculate uncertainty and update target's state
    Argument : 
        dt : time frame
        process_noise : noise matrix, include all noise for six components
        initial_states : initial states for all target
        initial_covariances : initial covariances for all target
    """
    def __init__(self, dt, process_noise, initial_states, initial_covariances):
        self.dt = dt
        self.process_noise = process_noise
        self.initial_states = initial_states
        self.initial_covariances = initial_covariances
        self.initial_aoi = np.zeros(len(self.initial_states))

        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

    def predict(self, target_id):
        """
        Perform EKF prediction for one target.
        Argument:
            target_id (int) : id of the target
        """
        state = self.states[target_id]
        covariance = self.covariances[target_id]
        predicted_state = self.F @ state
        predicted_covariance = (self.F @ covariance @ self.F.T + self.Q)

        self.states[target_id] = predicted_state
        self.covariances[target_id] = predicted_covariance

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

        state = self.states[target_id]
        covariance = self.covariances[target_id]

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
        measurement_dim = 3
        num_measurements = len(measurements)

        R_batch = np.zeros((measurement_dim * num_measurements, 
                            measurement_dim * num_measurements))

        for i, R in enumerate(covariance_list):
            start = i * measurement_dim
            end = start + measurement_dim

            R_batch[start:end, start:end] = R

        # Innovation
        innovation = (measurement_batch - predicted_measurement_batch)

        # Innovation covariance
        S = (H_batch @ covariance @ H_batch.T + R_batch)

        # Kalman gain
        K = (covariance @ H_batch.T @ np.linalg.inv(S))

        # State update
        updated_state = (state + K @ innovation)

        # Covariance update
        I = np.eye(6)

        updated_covariance = (I - K @ H_batch) @ covariance

        self.states[target_id] = updated_state
        self.covariances[target_id] = updated_covariance

    def update_aoi(self, target_id, detected):
        """
        Update AoI of one target.
        Argument :
            target_id (int) : target id
            detected (bool) : is the target id=target_id detected
        """
        if detected:
            self.aois[target_id] = 0.0
        else:
            self.aois[target_id] += self.dt