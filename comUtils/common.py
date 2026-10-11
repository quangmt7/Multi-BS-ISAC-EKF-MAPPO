"""
This file contain functions to use anywhere on this folder
"""
import numpy as np

def wrap_angle(angle):
    """
    Wrap an angle to the interval (-pi, pi].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Wrapped angle in (-pi, pi].
    """
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi

    if np.isclose(wrapped_angle, -np.pi):
        wrapped_angle = np.pi

    return wrapped_angle