import numpy as np
from comUtils.common import wrap_angle 


class BSMeasurement:
    """
    This class is in each BS, implementing the measurement based on the aggregated SINR
    Argument :
        ekf_config : (dict) configuration for EKF system
            state_dim : (int) number of state's dimension
            measurement_dim : (int) number of measurement's dimension
            gamma_0 : (float) reference SINR
            sigma_r_0 : (float) reference range variance
            sigma_theta_0 : (float) reference azumuth variance
            sigma_phi_0 : (float) reference elevation variance
            ap_theta_ratio : (float) ratio between aperture of the BS and reference on azimuth
            ap_phi_ratio : (float) ratio between aperture of the BS and reference on elevation
        bs_position : [float, float, float] position of BS
    """
    def __init__(self, bs_id, ekf_config, bs_position):

        self.bs_id = bs_id
        self.state_dim = ekf_config["state_dim"]
        self.measurement_dim = ekf_config["state_dim"]
        self.bs_position = bs_position

        self.gamma_0 = ekf_config["measurement_noise"]["gamma_0_linear"]

        self.sigma_r_0 = ekf_config["measurement_noise"]["sigma_range_0"]
        self.sigma_theta_0 = ekf_config["measurement_noise"]["sigma_azimuth_0_rad"]
        self.sigma_phi_0 = ekf_config["measurement_noise"]["sigma_elevation_0_rad"]

        self.ap_theta_ratio = ekf_config["measurement_noise"]["aperture"]["d_ap_theta_ratio"]
        self.ap_phi_ratio = ekf_config["measurement_noise"]["aperture"]["d_ap_phi_ratio"] 

    def measurement_function(self, target_state):
        """
        Convert the true target state into ideal range, azimuth, and elevation measurements.
        Argument: 
            target_state (float)[6] : state of the target
        Return:
            measurement (float)[3] : state of the target (with range, azimuth, elevation)
        """
        # Check the number of state's direction
        if (len(target_state) == self.state_dim):
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
        SINR must > 0 (= 0 -> not sense -> not calculate)
        Argument : 
            sensing_sinr (float) : effective SINR
        Return :
            R covariance (np.diag[3][3]) : covariance of the measurement.
        """
        if (sensing_sinr == 0):
            return ValueError(sensing_sinr)
        else :
            sinr_ratio = self.gamma_ref / sensing_sinr

            sigma_r_sq = (self.sigma_r_0 ** 2 * sinr_ratio)

            sigma_theta_sq = (self.sigma_theta_0 ** 2 * (sinr_ratio / self.ap_theta_ratio ** 2))

            sigma_phi_sq = (self.sigma_phi_0 ** 2 * (sinr_ratio / self.ap_phi_ratio ** 2))

            return np.diag([sigma_r_sq, sigma_theta_sq, sigma_phi_sq])
    
    def gen_meas(self, target_state, sensing_sinr):
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

        # Wrap azimuth and elevation to (-pi, pi].
        measurement[1] = wrap_angle(measurement[1])
        measurement[2] = wrap_angle(measurement[2])

        return measurement, R
    
        