"""Save and restore MAPPO and Hierarchical ISAC training checkpoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class CheckpointManager:
    """Persist and restore one MAPPO or Hierarchical ISAC training run.

    Argument:
        (config) (Mapping[str, Any]): Configuration associated with checkpoints.
        (ppo_agent) (Any): MAPPO agent whose state is persisted (``mode="mappo"``).
        (sac_agent) (Any): SAC frame-level agent (``mode="hierarchical"``).
        (dqn_agent) (Any): DQN subframe-level agent (``mode="hierarchical"``).
        (mode) (str | None): Checkpoint mode (``"mappo"``, ``"hierarchical"``,
            or ``None`` to auto-detect from provided agents).

    Return:
        (None).
    """

    SUPPORTED_MODES = ("mappo", "hierarchical")

    def __init__(
        self,
        config: Mapping[str, Any],
        ppo_agent: Any = None,
        *,
        sac_agent: Any = None,
        dqn_agent: Any = None,
        mode: str | None = None,
    ):
        """Store the checkpoint dependencies and validate the selected mode.

        Argument:
            (config) (Mapping[str, Any]): Validated project configuration.
            (ppo_agent) (Any): MAPPO agent instance for ``mode="mappo"``.
            (sac_agent) (Any): SAC agent instance for ``mode="hierarchical"``.
            (dqn_agent) (Any): DQN agent instance for ``mode="hierarchical"``.
            (mode) (str | None): Explicit checkpoint mode or ``None`` for
                auto-detection.

        Return:
            (None).
        """

        resolved_mode = self._resolve_mode(mode, ppo_agent, sac_agent, dqn_agent)
        self.config = dict(config)
        self.mode = resolved_mode
        self.ppo_agent = ppo_agent
        self.sac_agent = sac_agent
        self.dqn_agent = dqn_agent

    @classmethod
    def _resolve_mode(
        cls,
        mode: str | None,
        ppo_agent: Any,
        sac_agent: Any,
        dqn_agent: Any,
    ) -> str:
        """Validate mode and agent combinations fail-fast."""

        has_hier_agents = sac_agent is not None or dqn_agent is not None
        if mode is None:
            if ppo_agent is not None and has_hier_agents:
                raise ValueError(
                    "Cannot combine MAPPO 'ppo_agent' with hierarchical agents "
                    "('sac_agent' / 'dqn_agent')"
                )
            resolved_mode = "hierarchical" if has_hier_agents else "mappo"
        else:
            if mode not in cls.SUPPORTED_MODES:
                raise ValueError(
                    f"Unsupported checkpoint mode {mode!r}; "
                    f"expected one of {cls.SUPPORTED_MODES}"
                )
            resolved_mode = mode

        if resolved_mode == "mappo":
            if has_hier_agents:
                raise ValueError(
                    "MAPPO checkpoint mode does not accept 'sac_agent' or 'dqn_agent'"
                )
            if ppo_agent is None:
                raise ValueError("MAPPO checkpoint mode requires 'ppo_agent'")
        else:
            if ppo_agent is not None:
                raise ValueError(
                    "Hierarchical checkpoint mode does not accept 'ppo_agent'; "
                    "pass 'sac_agent' and 'dqn_agent'"
                )
            if sac_agent is None or dqn_agent is None:
                raise ValueError(
                    "Hierarchical checkpoint mode requires both 'sac_agent' and 'dqn_agent'"
                )
        return resolved_mode

    @staticmethod
    def _find_attr(obj: Any, candidates: tuple[str, ...]) -> Any:
        """Return the first non-None attribute among candidate names."""

        for attr_name in candidates:
            value = getattr(obj, attr_name, None)
            if value is not None:
                return value
        return None

    @classmethod
    def _get_state_dict_attr(
        cls, obj: Any, candidates: tuple[str, ...], label: str
    ) -> dict[str, Any]:
        """Extract ``state_dict()`` from a required component attribute."""

        component = cls._find_attr(obj, candidates)
        if component is None or not callable(getattr(component, "state_dict", None)):
            raise ValueError(
                f"Missing required {label} with state_dict() "
                f"(checked attributes: {', '.join(candidates)})"
            )
        return component.state_dict()

    @classmethod
    def _load_state_dict_attr(
        cls,
        obj: Any,
        candidates: tuple[str, ...],
        state_dict: Mapping[str, Any],
        label: str,
    ) -> None:
        """Restore ``load_state_dict()`` onto a required component attribute."""

        component = cls._find_attr(obj, candidates)
        if component is None or not callable(getattr(component, "load_state_dict", None)):
            raise ValueError(
                f"Missing required {label} with load_state_dict() "
                f"(checked attributes: {', '.join(candidates)})"
            )
        component.load_state_dict(state_dict)

    def _build_payload(self, episode: int, global_step: int) -> dict[str, Any]:
        """Build the full checkpoint payload for the active mode."""

        if isinstance(episode, bool) or not isinstance(episode, (int, np.integer)) or int(episode) < 0:
            raise ValueError("Checkpoint episode must be a non-negative integer")
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, (int, np.integer))
            or int(global_step) < 0
        ):
            raise ValueError("Checkpoint global_step must be a non-negative integer")

        payload: dict[str, Any] = {
            "episode": int(episode),
            "global_step": int(global_step),
            "config": self.config,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_random_state"] = torch.cuda.get_rng_state_all()

        if self.mode == "mappo":
            payload.update(
                {
                    "actor": self.ppo_agent.actornetwork.state_dict(),
                    "critic": self.ppo_agent.criticnetwork.state_dict(),
                    "actor_optimizer": self.ppo_agent.actor_optimizer.state_dict(),
                    "critic_optimizer": self.ppo_agent.critic_optimizer.state_dict(),
                }
            )
            return payload

        payload.update(
            {
                "sac_actor": self._get_state_dict_attr(
                    self.sac_agent, ("actor",), "SAC actor"
                ),
                "sac_critic_1": self._get_state_dict_attr(
                    self.sac_agent, ("critic_1",), "SAC critic_1"
                ),
                "sac_critic_2": self._get_state_dict_attr(
                    self.sac_agent, ("critic_2",), "SAC critic_2"
                ),
                "sac_actor_optimizer": self._get_state_dict_attr(
                    self.sac_agent,
                    ("actor_opt", "actor_optimizer"),
                    "SAC actor optimizer",
                ),
                "sac_critic_1_optimizer": self._get_state_dict_attr(
                    self.sac_agent,
                    ("critic_1_opt", "critic_1_optimizer"),
                    "SAC critic_1 optimizer",
                ),
                "sac_critic_2_optimizer": self._get_state_dict_attr(
                    self.sac_agent,
                    ("critic_2_opt", "critic_2_optimizer"),
                    "SAC critic_2 optimizer",
                ),
                "dqn_q_net": self._get_state_dict_attr(
                    self.dqn_agent, ("q_net",), "DQN q_net"
                ),
                "dqn_optimizer": self._get_state_dict_attr(
                    self.dqn_agent,
                    ("optimizer", "q_optimizer", "q_opt"),
                    "DQN optimizer",
                ),
            }
        )

        target_critic_1 = self._find_attr(self.sac_agent, ("target_critic_1",))
        if target_critic_1 is not None:
            payload["sac_target_critic_1"] = target_critic_1.state_dict()
        target_critic_2 = self._find_attr(self.sac_agent, ("target_critic_2",))
        if target_critic_2 is not None:
            payload["sac_target_critic_2"] = target_critic_2.state_dict()

        log_alpha = getattr(self.sac_agent, "log_alpha", None)
        if log_alpha is not None:
            if isinstance(log_alpha, torch.Tensor):
                payload["sac_log_alpha"] = log_alpha.detach().cpu().clone()
            elif callable(getattr(log_alpha, "state_dict", None)):
                payload["sac_log_alpha"] = log_alpha.state_dict()
            else:
                payload["sac_log_alpha"] = float(log_alpha)

        alpha_opt = self._find_attr(self.sac_agent, ("alpha_opt", "alpha_optimizer"))
        if alpha_opt is not None:
            payload["sac_alpha_optimizer"] = alpha_opt.state_dict()

        dqn_target = self._find_attr(
            self.dqn_agent, ("target_net", "q_target", "target_q_net")
        )
        if dqn_target is not None:
            payload["dqn_target_net"] = dqn_target.state_dict()

        dqn_eps = getattr(self.dqn_agent, "epsilon", None)
        if isinstance(dqn_eps, (int, float, np.integer, np.floating)) and not isinstance(
            dqn_eps, bool
        ):
            payload["dqn_epsilon"] = float(dqn_eps)

        return payload

    def save(self, path: str | Path, episode: int, global_step: int) -> None:
        """Atomically save model, optimizer and random-generator states.

        Argument:
            (path) (str | Path): Destination checkpoint path.
            (episode) (int): Zero-based episode index stored in the checkpoint.
            (global_step) (int): Number of environment steps already completed.

        Return:
            (None).
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        payload = self._build_payload(episode, global_step)

        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _resolve_load_device(self) -> Any:
        """Pick map_location device from the configured agent(s)."""

        if self.mode == "mappo":
            return getattr(self.ppo_agent, "device", torch.device("cpu"))
        sac_device = getattr(self.sac_agent, "device", None)
        if sac_device is not None:
            return sac_device
        dqn_device = getattr(self.dqn_agent, "device", None)
        if dqn_device is not None:
            return dqn_device
        return torch.device("cpu")

    def load(self, path: str | Path) -> tuple[int, int]:
        """Restore a compatible checkpoint.

        Argument:
            (path) (str | Path): Existing checkpoint path.

        Return:
            (progress) (tuple[int, int]): Next episode index and global step.

        Note:
            PyTorch checkpoints use pickle internally and must come from a trusted
            project run.
        """

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        device = self._resolve_load_device()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint root must be a mapping")

        common_required = (
            "episode",
            "global_step",
            "config",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
        )
        if self.mode == "mappo":
            mode_required = (
                "actor",
                "critic",
                "actor_optimizer",
                "critic_optimizer",
            )
        else:
            mode_required = (
                "sac_actor",
                "sac_critic_1",
                "sac_critic_2",
                "sac_actor_optimizer",
                "sac_critic_1_optimizer",
                "sac_critic_2_optimizer",
                "dqn_q_net",
                "dqn_optimizer",
            )

        required = common_required + mode_required
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"Checkpoint is missing: {', '.join(missing)}")
        if checkpoint["config"] != self.config:
            raise ValueError(
                "Checkpoint configuration differs from the current configuration"
            )

        raw_episode = checkpoint["episode"]
        raw_global_step = checkpoint["global_step"]
        if (
            isinstance(raw_episode, bool)
            or not isinstance(raw_episode, int)
            or raw_episode < 0
        ):
            raise ValueError("Checkpoint episode must be a non-negative integer")
        if (
            isinstance(raw_global_step, bool)
            or not isinstance(raw_global_step, int)
            or raw_global_step < 0
        ):
            raise ValueError("Checkpoint global_step must be a non-negative integer")

        if self.mode == "mappo":
            self.ppo_agent.actornetwork.load_state_dict(checkpoint["actor"])
            self.ppo_agent.criticnetwork.load_state_dict(checkpoint["critic"])
            self.ppo_agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.ppo_agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        else:
            self._load_state_dict_attr(
                self.sac_agent, ("actor",), checkpoint["sac_actor"], "SAC actor"
            )
            self._load_state_dict_attr(
                self.sac_agent,
                ("critic_1",),
                checkpoint["sac_critic_1"],
                "SAC critic_1",
            )
            self._load_state_dict_attr(
                self.sac_agent,
                ("critic_2",),
                checkpoint["sac_critic_2"],
                "SAC critic_2",
            )
            self._load_state_dict_attr(
                self.sac_agent,
                ("actor_opt", "actor_optimizer"),
                checkpoint["sac_actor_optimizer"],
                "SAC actor optimizer",
            )
            self._load_state_dict_attr(
                self.sac_agent,
                ("critic_1_opt", "critic_1_optimizer"),
                checkpoint["sac_critic_1_optimizer"],
                "SAC critic_1 optimizer",
            )
            self._load_state_dict_attr(
                self.sac_agent,
                ("critic_2_opt", "critic_2_optimizer"),
                checkpoint["sac_critic_2_optimizer"],
                "SAC critic_2 optimizer",
            )

            target_critic_1 = self._find_attr(self.sac_agent, ("target_critic_1",))
            if target_critic_1 is not None and "sac_target_critic_1" in checkpoint:
                target_critic_1.load_state_dict(checkpoint["sac_target_critic_1"])
            target_critic_2 = self._find_attr(self.sac_agent, ("target_critic_2",))
            if target_critic_2 is not None and "sac_target_critic_2" in checkpoint:
                target_critic_2.load_state_dict(checkpoint["sac_target_critic_2"])

            if "sac_log_alpha" in checkpoint and getattr(self.sac_agent, "log_alpha", None) is not None:
                saved_log_alpha = checkpoint["sac_log_alpha"]
                current_log_alpha = self.sac_agent.log_alpha
                if isinstance(current_log_alpha, torch.Tensor):
                    with torch.no_grad():
                        current_log_alpha.copy_(
                            torch.as_tensor(
                                saved_log_alpha,
                                dtype=current_log_alpha.dtype,
                                device=current_log_alpha.device,
                            )
                        )
                    if hasattr(self.sac_agent, "alpha"):
                        self.sac_agent.alpha = float(current_log_alpha.exp().item())
                elif callable(getattr(current_log_alpha, "load_state_dict", None)):
                    current_log_alpha.load_state_dict(saved_log_alpha)
                else:
                    self.sac_agent.log_alpha = float(saved_log_alpha)

            if "sac_alpha_optimizer" in checkpoint:
                alpha_opt = self._find_attr(
                    self.sac_agent, ("alpha_opt", "alpha_optimizer")
                )
                if alpha_opt is not None:
                    alpha_opt.load_state_dict(checkpoint["sac_alpha_optimizer"])

            self._load_state_dict_attr(
                self.dqn_agent, ("q_net",), checkpoint["dqn_q_net"], "DQN q_net"
            )
            self._load_state_dict_attr(
                self.dqn_agent,
                ("optimizer", "q_optimizer", "q_opt"),
                checkpoint["dqn_optimizer"],
                "DQN optimizer",
            )

            if "dqn_target_net" in checkpoint:
                dqn_target = self._find_attr(
                    self.dqn_agent, ("target_net", "q_target", "target_q_net")
                )
                if dqn_target is not None:
                    dqn_target.load_state_dict(checkpoint["dqn_target_net"])
            if "dqn_epsilon" in checkpoint and hasattr(self.dqn_agent, "epsilon"):
                self.dqn_agent.epsilon = float(checkpoint["dqn_epsilon"])

        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch_rng = checkpoint["torch_random_state"]
        torch.set_rng_state(
            torch_rng.cpu() if isinstance(torch_rng, torch.Tensor) else torch_rng
        )
        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
            cuda_rng = [
                state.cpu() if isinstance(state, torch.Tensor) else state
                for state in checkpoint["cuda_random_state"]
            ]
            torch.cuda.set_rng_state_all(cuda_rng)

        return raw_episode + 1, raw_global_step

    @staticmethod
    def prepare_metrics(path: str | Path, next_episode: int) -> str:
        """Align an existing metrics file with resumed checkpoint progress.

        Argument:
            (path) (str | Path): JSON Lines metrics path.
            (next_episode) (int): Zero-based episode index to run next.

        Return:
            (file_mode) (str): File mode used for the resumed training run.
        """

        path = Path(path)
        if next_episode == 0:
            return "w"
        if not path.exists():
            return "a"

        retained: list[str] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid metrics JSON at line {line_number}: {path}"
                    ) from exc
                episode = record.get("episode")
                if isinstance(episode, bool) or not isinstance(episode, int):
                    raise ValueError(
                        f"Invalid episode field at metrics line {line_number}: {path}"
                    )
                if episode <= next_episode:
                    retained.append(json.dumps(record, ensure_ascii=False))

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as stream:
                for line in retained:
                    stream.write(line + "\n")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return "a"