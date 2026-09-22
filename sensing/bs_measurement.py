import numpy as np


class BSMeasurement:
    """
    This class is in each BS, implementing the measurement based on the aggregated SINR
    Argument :
    bs_position : [float, float, float] position of BS
    gamma_ref : float (effective SINR (const for each BS))
    sigma_r_ref : float (variance of the range measurement at the reference condition)
    sigma_theta_ref : float (variance of the theta measurement at the reference condition)
    sigma_phi_ref : float (variance of the phi measurement at the reference condition)
    ap_theta_ratio : float (antenna aperture ratio for theta (of BS / reference))
    ap_phi_ratio : float (antenna aperture ratio for phi (of BS / reference))
    """
    def __init__(self, bs_position, gamma_ref, sigma_r_ref, sigma_theta_ref, sigma_phi_ref,
        ap_theta_ratio, ap_phi_ratio):
    
        self.bs_position = bs_position

        self.gamma_ref = gamma_ref

        self.sigma_r_ref = sigma_r_ref
        self.sigma_theta_ref = sigma_theta_ref
        self.sigma_phi_ref = sigma_phi_ref

        self.ap_theta_ratio = ap_theta_ratio
        self.ap_phi_ratio = ap_phi_ratio

    def measurement_function(self, target_state):
        """
        Convert the true target state into ideal range, azimuth, and elevation measurements.
        Argument: 
            target_state (float)[6] : state of the target
        Return:
            measurement (float)[3] : state of the target (with range, azimuth, elevation)
        """
        x, y, z = target_state[:3]
        xb, yb, zb = self.bs_position
        dx = x - xb
        dy = y - yb
        dz = z - zb

        horizontal_distance = np.sqrt(dx**2 + dy**2)
        range_ = np.sqrt(dx**2 + dy**2 + dz**2)
        azimuth = np.arctan2(dy, dx)
        elevation = np.arctan2(dz, horizontal_distance)

        return np.array([range_, azimuth, elevation])

    def measurement_covariance(self, sensing_sinr):
        """
        Return measurement noise covariance R of the measurement.
        Argument : 
            sensing_sinr (float) : effective SINR
        Return :
            R covariance (np.diag[3][3]) : covariance of the measurement.
        """
        sinr_ratio = self.gamma_ref / sensing_sinr

        sigma_r_sq = (self.sigma_r_ref ** 2 * sinr_ratio)

        sigma_theta_sq = (self.sigma_theta_ref ** 2 * (sinr_ratio / self.ap_theta_ratio ** 2))

        sigma_phi_sq = (self.sigma_phi_ref ** 2 * (sinr_ratio / self.ap_phi_ratio ** 2))

        return np.diag([sigma_r_sq, sigma_theta_sq, sigma_phi_sq])
    
    def generate_measurement(self, target_state, sensing_sinr):
        """
        Generate a noisy sensing measurement.
        Argument : 
            target_state (float)[6] : state of the target
            sensing_sinr (float) : effective SINR
        Return : 
            measurement (float)[3] : measurement result (with noise)
            R (float)[3][3] : covariance matrix
        """
        true_measurement = self.measurement_function(target_state)

        R = self.measurement_covariance(sensing_sinr)

        noise = np.random.multivariate_normal(mean=np.zeros(3),cov=R)

        measurement = true_measurement + noise

        return measurement, R
    
        