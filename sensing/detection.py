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
        # Set of successed sensing target
        self.sense_suc = []

    def detect_prob(self, sensing_sinr):
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
        p_d = self.detect_prob(sensing_sinr)
        detected = np.random.random() < p_d
        return detected, p_d

