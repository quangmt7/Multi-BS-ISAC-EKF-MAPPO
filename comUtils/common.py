"""
This file contain functions to use anywhere on this folder
"""
import numpy as np
import torch.nn as nn
import math

def wrap_angle(angle):
    """
    Wrap an angle to the interval (-pi, pi].

    Parameters
    ----------
    angle : float or np.ndarray
        Angle or angles in radians.

    Returns
    -------
    float or np.ndarray
        Wrapped angle or angles in (-pi, pi].
    """
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi

    wrapped_angle = np.where(np.isclose(wrapped_angle, -np.pi), np.pi, wrapped_angle)
    return float(wrapped_angle) if np.ndim(wrapped_angle) == 0 else wrapped_angle


def get_activation(name):
    """
    Create an activation module.

    Args:
        name (str): Activation function name, e.g. "tanh" or "relu".

    Returns:
        nn.Module: Activation module.
    """
    name = name.lower()

    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim,
    hiddens_dim,
    activation="tanh",
    use_layer_norm=True,
    orthogonal_init=True,
    gain=0.01,
    isActor=True,
):
    """
    Build a shared MLP backbone for Actor or Critic.

    Args:
        input_dim (int): Dimension of the input vector.
        hiddens_dim (list[int]): Hidden-layer sizes, e.g. [256, 256].
        activation (str): Hidden-layer activation function.
        use_layer_norm (bool): Whether to use Layer Normalization.
        orthogonal_init (bool): Whether to use Orthogonal Initialization.
        gain (float): Initialization gain for the output head.
        isActor (bool): True for Actor, False for Critic.

    Returns:
        nn.Sequential: MLP backbone.
        float: Recommended initialization gain for the output head.
    """
    if len(hiddens_dim) != 2:
        raise ValueError("hiddens_dim must contain exactly two values.")

    layers = []
    current_dim = input_dim

    for hidden_dim in hiddens_dim:
        linear = nn.Linear(current_dim, hidden_dim)

        if orthogonal_init:
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
            nn.init.constant_(linear.bias, 0.0)

        layers.append(linear)

        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(get_activation(activation))
        current_dim = hidden_dim

    backbone = nn.Sequential(*layers)

    # Actor uses the configured small gain.
    # Critic uses gain 1.0 for its output layer.
    output_gain = gain if isActor else 1.0

    return backbone, output_gain
