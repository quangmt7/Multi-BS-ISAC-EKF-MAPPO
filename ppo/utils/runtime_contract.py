"""Validate runtime data exchanged during MAPPO and Hierarchical ISAC training."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


class RuntimeContract:
    """Validate data exchanged by the environment, agents, and buffers.

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

        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        system = config.get("system_parameters")
        if not isinstance(system, Mapping):
            raise ValueError("Missing or invalid config section: system_parameters")

        dims: dict[str, int] = {}
        for key in (
            "num_bs",
            "num_target",
            "num_embb",
            "num_urllc",
            "num_ue_1embb",
            "num_ue_1urllc",
        ):
            val = system.get(key)
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) <= 0:
                raise ValueError(f"system_parameters.{key} must be a positive integer")
            dims[key] = int(val)

        n_prb = system.get("nPrb")
        if not isinstance(n_prb, (list, tuple)) or not n_prb:
            raise ValueError("system_parameters.nPrb must be a non-empty list of positive integers")
        total_prb = 0
        for idx, count in enumerate(n_prb):
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or int(count) <= 0
            ):
                raise ValueError(f"system_parameters.nPrb[{idx}] must be a positive integer")
            total_prb += int(count)

        self.num_bs = dims["num_bs"]
        self.num_prb = total_prb  # flat policy order: BWP then PRB
        num_ue = (
            dims["num_embb"] * dims["num_ue_1embb"]
            + dims["num_urllc"] * dims["num_ue_1urllc"]
        )
        self.max_destination = int(dims["num_target"] + num_ue)

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

        local_array = self._to_float32_array(local_observations, "local observations")
        global_array = self._to_float32_array(global_state, "global state")

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

        reward_val = self._finite_scalar(
            reward, "env.step() must return one finite scalar team reward"
        )
        if not isinstance(info, Mapping):
            raise ValueError("env.step() info must be a mapping")

        return (
            np.asarray(local_obs, dtype=np.float32),
            np.asarray(global_state, dtype=np.float32),
            reward_val,
            self._boolean_flag(terminated, "terminated"),
            self._boolean_flag(truncated, "truncated"),
            dict(info),
        )

    def hier_step_result(
        self,
        result: Any,
        expected_local_dim: int,
        expected_global_dim: int,
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        """Validate one hierarchical environment transition 6-tuple.

        Argument:
            (result) (Any): Raw value returned by ``env.step``.
            (expected_local_dim) (int): Local state dimension fixed at reset.
            (expected_global_dim) (int): Global state dimension fixed at reset.

        Return:
            (transition) (tuple): Validated next states, reward, flags and info.
        """

        return self.step_result(result, expected_local_dim, expected_global_dim)

    def _validate_prb_actions(self, actions_prb: Any) -> np.ndarray:
        """Validate discrete PRB allocation indices across all BSs."""

        prb_array = np.asarray(actions_prb)
        expected_shape = (self.num_bs, self.num_prb)
        if prb_array.shape != expected_shape:
            raise ValueError(
                f"PRB action shape must be {expected_shape}, got {prb_array.shape}"
            )
        if not np.issubdtype(prb_array.dtype, np.integer):
            raise ValueError("PRB actions must use an integer dtype")
        if np.any(prb_array < 0) or np.any(prb_array > self.max_destination):
            raise ValueError(f"PRB actions must be in [0, {self.max_destination}]")
        return prb_array

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

        actions_prb = self._validate_prb_actions(output[0])
        actions_pwr = self._to_float32_array(output[1], "power actions")
        log_probs = self._to_float32_array(output[2], "log-probabilities")
        value = self._finite_scalar(
            output[3], "centralized critic value must be one finite scalar"
        )
        expected_shape = (self.num_bs, self.num_prb)

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

        return actions_prb, actions_pwr, log_probs, value

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

        for method_name in ("select_action", "estimate_value", "update"):
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

    def sac_state(self, state: Any) -> tuple[np.ndarray, int]:
        """Validate one frame-level SAC state vector and return it with its size.

        Argument:
            (state) (Any): Candidate frame-level SAC state vector.

        Return:
            (validated) (tuple[np.ndarray, int]): Float32 1D array and dimension.
        """

        state_array = self._finite_1d_float32(state, "SAC state")
        return state_array, int(state_array.size)

    def sac_action(
        self, action: Any, expected_dim: int | None = None
    ) -> np.ndarray:
        """Validate one flattened SAC continuous action vector.

        Argument:
            (action) (Any): Candidate SAC action vector.
            (expected_dim) (int | None): Optional expected flattened length.

        Return:
            (validated) (np.ndarray): Float32 1D non-negative finite action vector.
        """

        action_array = self._finite_1d_float32(action, "SAC action")
        if expected_dim is not None and action_array.size != int(expected_dim):
            raise ValueError(
                f"SAC action dimension mismatch: expected {int(expected_dim)}, "
                f"got {action_array.size}"
            )
        if np.any(action_array < 0.0):
            raise ValueError("SAC action must contain only non-negative values")
        return action_array

    def dqn_actions(self, actions_prb: Any) -> np.ndarray:
        """Validate discrete DQN PRB actions for all base stations.

        Argument:
            (actions_prb) (Any): Candidate discrete PRB assignments.

        Return:
            (validated) (np.ndarray): Validated integer PRB action array.
        """

        return self._validate_prb_actions(actions_prb)

    def sac_agent_structure(self, sac_agent: Any, sac_state_dim: int) -> None:
        """Validate the SAC agent API, networks, optimizers, and input dimensions.

        Argument:
            (sac_agent) (Any): Frame-level SAC agent instance.
            (sac_state_dim) (int): Frame-level SAC state dimension.

        Return:
            (None).
        """

        if sac_agent is None:
            raise TypeError("SAC agent must not be None")

        for method_name in ("select_action", "update"):
            if not callable(getattr(sac_agent, method_name, None)):
                raise TypeError(f"SAC agent must provide a callable {method_name}()")

        if getattr(sac_agent, "replay_buffer", None) is None:
            raise TypeError("SAC agent is missing required attribute: replay_buffer")

        for attr_name in ("actor", "critic_1", "critic_2"):
            if getattr(sac_agent, attr_name, None) is None:
                raise TypeError(f"SAC agent is missing required attribute: {attr_name}")

        for opt_candidates in (
            ("actor_opt", "actor_optimizer"),
            ("critic_1_opt", "critic_1_optimizer"),
            ("critic_2_opt", "critic_2_optimizer"),
        ):
            if not any(getattr(sac_agent, name, None) is not None for name in opt_candidates):
                raise TypeError(
                    f"SAC agent is missing required optimizer attribute: {opt_candidates[0]}"
                )

        actor_input = self._first_linear_input(sac_agent.actor)
        if actor_input is not None and actor_input != sac_state_dim:
            raise ValueError(
                f"SAC actor expects {actor_input} inputs, but env returns {sac_state_dim}"
            )

        action_dim = getattr(sac_agent, "action_dim", None)
        if isinstance(action_dim, (int, np.integer)) and not isinstance(action_dim, bool):
            expected_q_dim = int(sac_state_dim) + int(action_dim)
            for critic_name in ("critic_1", "critic_2"):
                critic_input = self._first_linear_input(getattr(sac_agent, critic_name))
                if critic_input is not None and critic_input != expected_q_dim:
                    raise ValueError(
                        f"SAC {critic_name} expects {critic_input} inputs, "
                        f"but state_dim + action_dim is {expected_q_dim}"
                    )

    def dqn_agent_structure(
        self, dqn_agent: Any, dqn_state_dim: int | None = None
    ) -> None:
        """Validate the DQN agent API, Q-network, optimizer, and replay buffer.

        Argument:
            (dqn_agent) (Any): Subframe-level DQN agent instance.
            (dqn_state_dim) (int | None): Optional DQN state dimension check.

        Return:
            (None).
        """

        if dqn_agent is None:
            raise TypeError("DQN agent must not be None")

        if not (
            callable(getattr(dqn_agent, "select_action", None))
            or callable(getattr(dqn_agent, "select_actions_all_bs", None))
        ):
            raise TypeError(
                "DQN agent must provide a callable select_action() or select_actions_all_bs()"
            )
        if not callable(getattr(dqn_agent, "update", None)):
            raise TypeError("DQN agent must provide a callable update()")

        for attr_name in ("replay_buffer", "q_net"):
            if getattr(dqn_agent, attr_name, None) is None:
                raise TypeError(f"DQN agent is missing required attribute: {attr_name}")

        if not any(
            getattr(dqn_agent, name, None) is not None
            for name in ("optimizer", "q_optimizer", "q_opt")
        ):
            raise TypeError(
                "DQN agent is missing required optimizer attribute (optimizer or q_optimizer)"
            )

        if dqn_state_dim is not None:
            q_input = self._first_linear_input(dqn_agent.q_net)
            if q_input is not None and q_input != int(dqn_state_dim):
                raise ValueError(
                    f"DQN q_net expects {q_input} inputs, but env returns {dqn_state_dim}"
                )

    def finite_agent_parameters(self, agent: Any) -> None:
        """Reject an update that creates non-finite neural-network parameters.

        Argument:
            (agent) (Any): MAPPO agent instance after an optimization update.

        Return:
            (None).
        """

        for network_name in ("actornetwork", "criticnetwork"):
            self._assert_finite_module_parameters(
                getattr(agent, network_name, None), network_name
            )

    def finite_hier_parameters(self, sac_agent: Any, dqn_agent: Any) -> None:
        """Reject an update that creates non-finite SAC or DQN parameters.

        Argument:
            (sac_agent) (Any): Frame-level SAC agent instance.
            (dqn_agent) (Any): Subframe-level DQN agent instance.

        Return:
            (None).
        """

        if sac_agent is None:
            raise TypeError("SAC agent must not be None")
        if dqn_agent is None:
            raise TypeError("DQN agent must not be None")

        for network_name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            self._assert_finite_module_parameters(
                getattr(sac_agent, network_name, None), f"sac_agent.{network_name}"
            )

        log_alpha = getattr(sac_agent, "log_alpha", None)
        if isinstance(log_alpha, torch.Tensor) and not torch.isfinite(log_alpha).all():
            raise FloatingPointError("sac_agent.log_alpha contains NaN or Inf")

        for network_name in ("q_net", "target_net", "q_target", "target_q_net"):
            self._assert_finite_module_parameters(
                getattr(dqn_agent, network_name, None), f"dqn_agent.{network_name}"
            )

    @staticmethod
    def _assert_finite_module_parameters(network: Any, label: str) -> None:
        """Raise FloatingPointError if any parameter in a PyTorch module is non-finite."""

        if not isinstance(network, torch.nn.Module):
            return
        for parameter_name, parameter in network.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    f"{label}.{parameter_name} contains NaN or Inf"
                )

    @classmethod
    def _contains_bool(cls, value: Any) -> bool:
        """Detect Python or NumPy booleans, including inside nested sequences."""

        if isinstance(value, (bool, np.bool_)):
            return True
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                return True
            if value.dtype == object:
                return any(cls._contains_bool(item) for item in value.flat)
            return False
        if isinstance(value, (list, tuple)):
            return any(cls._contains_bool(item) for item in value)
        return False

    @classmethod
    def _to_float32_array(cls, value: Any, label: str) -> np.ndarray:
        """Convert numeric input to float32 while rejecting booleans."""

        if cls._contains_bool(value):
            raise ValueError(f"{label} must be numeric, not boolean")
        raw = np.asarray(value)
        if raw.dtype == np.bool_:
            raise ValueError(f"{label} must be numeric, not boolean")
        try:
            return np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric") from exc

    @classmethod
    def _finite_scalar(cls, value: Any, message: str) -> float:
        """Convert one non-boolean scalar to a finite Python float."""

        if cls._contains_bool(value):
            raise ValueError(message)
        arr = np.asarray(value)
        if arr.ndim != 0 or arr.dtype == np.bool_:
            raise ValueError(message)
        try:
            scalar = float(arr.item())
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(scalar):
            raise ValueError(message)
        return scalar

    @classmethod
    def _finite_1d_float32(cls, value: Any, label: str) -> np.ndarray:
        """Validate a non-empty finite 1D float32 vector."""

        array = cls._to_float32_array(value, label)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{label} must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain only finite values")
        return array

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
