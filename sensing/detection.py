import numpy as np
from scipy.special import erfc, erfcinv


class Detection:
    """
    Perform detection for each BS-target pair and select successful measurements.

    Parameters
    ----------
    false_alarm_probability : float
        False-alarm probability P_F. Must satisfy 0 < P_F < 1.
    """

    def __init__(self, false_alarm_probability=1e-6):
        """
        Initialize the detection model.

        Parameters
        ----------
        false_alarm_probability : float
            Probability of false alarm P_F, used to calculate P_D.
        """
        if not 0 < false_alarm_probability < 1:
            raise ValueError("false_alarm_probability must be between 0 and 1.")

        self.false_alarm_probability = false_alarm_probability

    def detection_probability(self, sensing_sinr):
        """
        Calculate the detection probability P_D for one BS-target pair.

        Parameters
        ----------
        sensing_sinr : float
            Sensing SINR in linear scale (not dB). Must be non-negative.

        Returns
        -------
        float
            Detection probability P_D in [0, 1].
        """
        if sensing_sinr < 0:
            raise ValueError("sensing_sinr must be non-negative.")

        p_f = self.false_alarm_probability
        p_d = 0.5 * erfc(erfcinv(2 * p_f) - np.sqrt(sensing_sinr))
        return float(np.clip(p_d, 0.0, 1.0))

    def detect(self, sensing_sinr):
        """
        Decide whether one BS detects one target.

        Parameters
        ----------
        sensing_sinr : float
            Sensing SINR in linear scale (not dB).

        Returns
        -------
        detected : bool
            True if the target is detected; otherwise False.
        p_d : float
            Detection probability used for this decision.
        """
        p_d = self.detection_probability(sensing_sinr)
        detected = np.random.random() < p_d
        return detected, p_d

    def collect_measurements(
        self,
        target_state,
        sensing_sinr_by_bs,
        measurement_models,
    ):
        """
        Detect the target at each BS and collect successful measurements.

        Parameters
        ----------
        target_state : array-like, shape (6,)
            True target state [x, y, z, vx, vy, vz] used by the simulator
            to generate synthetic measurements.
        sensing_sinr_by_bs : dict
            Dictionary {bs_id: sensing_sinr}. SINR must be linear, not dB.
        measurement_models : dict
            Dictionary {bs_id: BSMeasurement object}. Each object must provide
            bs_position and generate_measurement(target_state, sensing_sinr).

        Returns
        -------
        measurements : list
            Successful measurements in the format expected by EKF.update:
            [(bs_position, measurement, R), ...].
        detection_results : dict
            Dictionary {bs_id: {"detected": bool, "P_D": float}}.
        """
        measurements = []
        detection_results = {}

        for bs_id, sensing_sinr in sensing_sinr_by_bs.items():
            if bs_id not in measurement_models:
                raise KeyError("Missing measurement model for BS {}".format(bs_id))

            detected, p_d = self.detect(sensing_sinr)
            detection_results[bs_id] = {
                "detected": detected,
                "P_D": p_d,
            }

            if detected:
                measurement_model = measurement_models[bs_id]
                measurement, R = measurement_model.generate_measurement(
                    target_state, sensing_sinr
                )
                measurements.append(
                    (measurement_model.bs_position, measurement, R)
                )

        return measurements, detection_results
