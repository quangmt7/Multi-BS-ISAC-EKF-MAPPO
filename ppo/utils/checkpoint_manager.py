"""Save and restore MAPPO training checkpoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class CheckpointManager:
    """Persist and restore one MAPPO training run.

    Argument:
        (config) (Mapping[str, Any]): Configuration associated with checkpoints.
        (agent) (Any): MAPPO agent whose state is persisted.

    Return:
        (None).
    """

    def __init__(self, config: Mapping[str, Any], agent: Any):
        """Store the checkpoint dependencies.

        Argument:
            (config) (Mapping[str, Any]): Validated project configuration.
            (agent) (Any): MAPPO agent instance.

        Return:
            (None).
        """

        self.config = dict(config)
        self.agent = agent

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
        payload = {
            "episode": int(episode),
            "global_step": int(global_step),
            "config": self.config,
            "actor": self.agent.actornetwork.state_dict(),
            "critic": self.agent.criticnetwork.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer": self.agent.critic_optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_random_state"] = torch.cuda.get_rng_state_all()

        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

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

        device = getattr(self.agent, "device", torch.device("cpu"))
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint root must be a mapping")

        required = (
            "episode",
            "global_step",
            "config",
            "actor",
            "critic",
            "actor_optimizer",
            "critic_optimizer",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
        )
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

        self.agent.actornetwork.load_state_dict(checkpoint["actor"])
        self.agent.criticnetwork.load_state_dict(checkpoint["critic"])
        self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

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