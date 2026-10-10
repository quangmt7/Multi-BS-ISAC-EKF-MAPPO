"""Validate runtime data exchanged during MAPPO training."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


class RuntimeContract:
    """Validate data exchanged by the environment, agent and rollout buffer.

    Argument:
        (config) (Mapping[str, Any]): Validated project configuration.

    Return:
        (None).
    """

    def __init__(self, config: Mapping[str, Any]):
        """Store dimensions and valid action ranges from the configuration.

        Argument:
            (config) (Mapping[str, Any]): Validated project configuration.

        Return:
            (None).
        """

        system = config["system_parameters"]
        self.num_bs = int(system["num_bs"])
        self.num_prb = int(system["num_prb"])
        self.max_destination = int(system["num_target"] + system["num_ue"])

    def observation_dimensions(
        self, local_observations: Any, global_state: Any
    ) -> tuple[int, int]:
        """Validate observations and return their dimensions.

        Argument:
            (local_observations) (Any)[num_bs, local_obs_dim]: Local BS states.
            (global_state) (Any)[global_state_dim]: Centralized critic state.

        Return:
            (dimensions) (tuple[int, int]): Local and global state dimensions.
        """

        local_array = np.asarray(local_observations, dtype=np.float32)
        global_array = np.asarray(global_state, dtype=np.float32)

        if local_array.ndim != 2:
            raise ValueError(
                "local observations must have shape [num_bs, local_obs_dim]"
            )
        if local_array.shape[0] != self.num_bs:
            raise ValueError(
                f"local observations contain {local_array.shape[0]} agents; "
                f"config num_bs is {self.num_bs}"
            )
        if local_array.shape[1] <= 0:
            raise ValueError("local_obs_dim must be positive")
        if global_array.ndim != 1 or global_array.size == 0:
            raise ValueError("global state must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(local_array)) or not np.all(
            np.isfinite(global_array)
        ):
            raise ValueError("observations must contain only finite values")

        return int(local_array.shape[1]), int(global_array.size)

    def step_result(
        self,
        result: Any,
        expected_local_dim: int,
        expected_global_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        """Validate one transition returned by the environment.

        Argument:
            (result) (Any): Raw value returned by ``env.step``.
            (expected_local_dim) (int): Local state dimension fixed at reset.
            (expected_global_dim) (int): Global state dimension fixed at reset.

        Return:
            (transition) (tuple): Validated next states, reward, flags and info.
        """

        if not isinstance(result, tuple) or len(result) != 6:
            raise ValueError(
                "env.step() must return (local_obs, global_state, reward, "
                "terminated, truncated, info)"
            )

        local_obs, global_state, reward, terminated, truncated, info = result
        actual_local_dim, actual_global_dim = self.observation_dimensions(
            local_obs, global_state
        )
        if (
            actual_local_dim != expected_local_dim
            or actual_global_dim != expected_global_dim
        ):
            raise ValueError(
                "observation dimensions changed during the episode: "
                f"expected ({expected_local_dim}, {expected_global_dim}), "
                f"received ({actual_local_dim}, {actual_global_dim})"
            )

        reward_array = np.asarray(reward)
        if reward_array.ndim != 0 or not np.isfinite(reward_array.item()):
            raise ValueError("env.step() must return one finite scalar team reward")
        if not isinstance(info, Mapping):
            raise ValueError("env.step() info must be a mapping")

        return (
            np.asarray(local_obs, dtype=np.float32),
            np.asarray(global_state, dtype=np.float32),
            float(reward_array.item()),
            self._boolean_flag(terminated, "terminated"),
            self._boolean_flag(truncated, "truncated"),
            dict(info),
        )

    def policy_output(
        self, output: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Validate actions, log-probabilities and value from the agent.

        Argument:
            (output) (Any): Value returned by ``agent.select_action``.

        Return:
            (policy_output) (tuple): Validated PRB actions, power, log-probability
                and centralized value.
        """

        if not isinstance(output, tuple) or len(output) != 4:
            raise ValueError(
                "agent.select_action() must return "
                "(actions_prb, actions_pwr, log_probs, value)"
            )

        actions_prb = np.asarray(output[0])
        actions_pwr = np.asarray(output[1], dtype=np.float32)
        log_probs = np.asarray(output[2], dtype=np.float32)
        value_array = np.asarray(output[3])
        expected_shape = (self.num_bs, self.num_prb)

        if actions_prb.shape != expected_shape:
            raise ValueError(
                f"PRB action shape must be {expected_shape}, got {actions_prb.shape}"
            )
        if not np.issubdtype(actions_prb.dtype, np.integer):
            raise ValueError("PRB actions must use an integer dtype")
        if np.any(actions_prb < 0) or np.any(actions_prb > self.max_destination):
            raise ValueError(f"PRB actions must be in [0, {self.max_destination}]")
        if actions_pwr.shape != expected_shape:
            raise ValueError(
                f"power action shape must be {expected_shape}, got {actions_pwr.shape}"
            )
        if not np.all(np.isfinite(actions_pwr)):
            raise ValueError("power actions contain NaN or Inf")
        if np.any(actions_pwr < 0.0) or np.any(actions_pwr > 1.0):
            raise ValueError(
                "power actions must be normalized to [0, 1] before action mapping"
            )
        if log_probs.shape != (self.num_bs,) or not np.all(np.isfinite(log_probs)):
            raise ValueError(
                f"log-probabilities must be finite with shape ({self.num_bs},)"
            )
        if value_array.ndim != 0 or not np.isfinite(value_array.item()):
            raise ValueError("centralized critic value must be one finite scalar")

        return actions_prb, actions_pwr, log_probs, float(value_array.item())

    def agent_structure(
        self, agent: Any, local_obs_dim: int, global_state_dim: int
    ) -> None:
        """Validate the agent API and neural-network input dimensions.

        Argument:
            (agent) (Any): MAPPO agent instance.
            (local_obs_dim) (int): Local observation dimension from the environment.
            (global_state_dim) (int): Global state dimension from the environment.

        Return:
            (None).
        """

        for method_name in ("select_action", "update"):
            if not callable(getattr(agent, method_name, None)):
                raise TypeError(f"Agent must provide a callable {method_name}()")

        for attribute in (
            "actornetwork",
            "criticnetwork",
            "actor_optimizer",
            "critic_optimizer",
        ):
            if not hasattr(agent, attribute):
                raise TypeError(f"Agent is missing required attribute: {attribute}")

        actor_input = self._first_linear_input(agent.actornetwork)
        critic_input = self._first_linear_input(agent.criticnetwork)
        if actor_input is not None and actor_input != local_obs_dim:
            raise ValueError(
                f"Actor expects {actor_input} inputs, but env returns {local_obs_dim}"
            )
        if critic_input is not None and critic_input != global_state_dim:
            raise ValueError(
                f"Critic expects {critic_input} inputs, but env returns {global_state_dim}"
            )

    def finite_agent_parameters(self, agent: Any) -> None:
        """Reject an update that creates non-finite neural-network parameters.

        Argument:
            (agent) (Any): MAPPO agent instance after an optimization update.

        Return:
            (None).
        """

        for network_name in ("actornetwork", "criticnetwork"):
            network = getattr(agent, network_name, None)
            if not isinstance(network, torch.nn.Module):
                continue
            for parameter_name, parameter in network.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"{network_name}.{parameter_name} contains NaN or Inf"
                    )

    @staticmethod
    def _boolean_flag(value: Any, name: str) -> bool:
        """Convert one scalar Boolean flag without accepting ambiguous values.

        Argument:
            (value) (Any): Candidate Boolean value.
            (name) (str): Field name used in an error message.

        Return:
            (flag) (bool): Validated Boolean value.
        """

        value_array = np.asarray(value)
        if value_array.ndim != 0 or not isinstance(
            value_array.item(), (bool, np.bool_)
        ):
            raise ValueError(f"env.step() {name} must be one Boolean scalar")
        return bool(value_array.item())

    @staticmethod
    def _first_linear_input(network: Any) -> int | None:
        """Read the first linear-layer input dimension of a network.

        Argument:
            (network) (Any): Candidate PyTorch network.

        Return:
            (input_dim) (int | None): Input size, or None for a non-PyTorch object.
        """

        if not isinstance(network, torch.nn.Module):
            return None
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        return None