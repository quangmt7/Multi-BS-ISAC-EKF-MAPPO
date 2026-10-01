"""Training entry point for the multi-BS ISAC MAPPO baseline.

The training loop only coordinates the environment, rollout buffer and agent.
State construction, action mapping, physical models and reward calculation remain
owned by the environment, as required by the project architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml


CONFIG_FILES = ("system.yaml", "ekf.yaml", "mappo.yaml")


def load_config(config_dir: str | Path) -> dict[str, Any]:
    """Load and merge the three project configuration files.

    Each file must own a different top-level section. Rejecting duplicate sections
    prevents one file from silently overriding another.
    """

    config_dir = Path(config_dir)
    merged: dict[str, Any] = {}

    for filename in CONFIG_FILES:
        path = config_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing configuration file: {path}")

        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}

        if not isinstance(document, dict):
            raise ValueError(f"{path} must contain a YAML mapping at its root")

        duplicates = merged.keys() & document.keys()
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"Duplicate top-level config section(s): {names}")
        merged.update(document)

    validate_config(merged)
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate fields that are required by the training loop and agent contract."""

    for section in ("system_parameters", "ekf_parameters", "mappo_parameters"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"Missing or invalid config section: {section}")

    system = config["system_parameters"]
    for name in ("num_bs", "num_ue", "num_target", "num_prb"):
        value = system.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"system_parameters.{name} must be a positive integer")

    training = config["mappo_parameters"].get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Missing or invalid config section: mappo_parameters.training")

    for name in ("total_episodes", "episode_length"):
        value = training.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"mappo_parameters.training.{name} must be a positive integer")

    ppo = config["mappo_parameters"].get("ppo", {})
    if ppo.get("use_clipped_value_loss", False):
        raise ValueError(
            "Baseline requires MSE critic loss without clipped value loss; "
            "set mappo_parameters.ppo.use_clipped_value_loss to false"
        )
    if ppo.get("use_huber_loss", False):
        raise ValueError(
            "Baseline requires MSE critic loss; "
            "set mappo_parameters.ppo.use_huber_loss to false"
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch without forcing deterministic algorithms."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_dimensions(
    local_observations: Any, global_state: Any, num_bs: int
) -> tuple[int, int]:
    """Validate reset output and infer dimensions from the environment.

    Dimensions are intentionally inferred instead of copied from ``mappo.yaml``.
    This keeps ``train.py`` compatible while the group finalizes the normalized
    State definition.
    """

    local_array = np.asarray(local_observations, dtype=np.float32)
    global_array = np.asarray(global_state, dtype=np.float32)

    if local_array.ndim != 2:
        raise ValueError(
            "local observations must have shape [num_bs, local_obs_dim]"
        )
    if local_array.shape[0] != num_bs:
        raise ValueError(
            f"local observations contain {local_array.shape[0]} agents; "
            f"config num_bs is {num_bs}"
        )
    if local_array.shape[1] <= 0:
        raise ValueError("local_obs_dim must be positive")
    if global_array.ndim != 1 or global_array.size == 0:
        raise ValueError("global state must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(local_array)) or not np.all(np.isfinite(global_array)):
        raise ValueError("observations must contain only finite values")

    return int(local_array.shape[1]), int(global_array.size)


def _reset_environment(env: Any, seed: int) -> tuple[np.ndarray, np.ndarray]:
    result = env.reset(seed=seed)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("env.reset() must return (local_observations, global_state)")
    return np.asarray(result[0], dtype=np.float32), np.asarray(result[1], dtype=np.float32)


def _validate_step_result(
    result: Any, num_bs: int, local_obs_dim: int, global_state_dim: int
) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or len(result) != 6:
        raise ValueError(
            "env.step() must return (local_obs, global_state, reward, "
            "terminated, truncated, info)"
        )

    local_obs, global_state, reward, terminated, truncated, info = result
    actual_local_dim, actual_global_dim = infer_dimensions(
        local_obs, global_state, num_bs
    )
    if actual_local_dim != local_obs_dim or actual_global_dim != global_state_dim:
        raise ValueError(
            "observation dimensions changed during the episode: "
            f"expected ({local_obs_dim}, {global_state_dim}), "
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
        bool(terminated),
        bool(truncated),
        dict(info),
    )


def _default_component_types() -> tuple[type[Any], type[Any]]:
    """Import the current project agent and rollout buffer lazily.

    Lazy imports keep config validation and unit tests independent of unfinished
    environment modules on other branches.
    """

    try:
        from ppo.agent import MAPPOAgent, MultiAgentRolloutBuffer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "ppo.agent is unavailable on this branch. Merge the reviewed agent "
            "implementation before starting real training."
        ) from exc

    return MAPPOAgent, MultiAgentRolloutBuffer


def _validate_policy_output(
    actions_prb: Any,
    actions_pwr: Any,
    log_probs: Any,
    value: Any,
    *,
    num_bs: int,
    num_prb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    actions_prb = np.asarray(actions_prb)
    actions_pwr = np.asarray(actions_pwr, dtype=np.float32)
    log_probs = np.asarray(log_probs, dtype=np.float32)
    value_array = np.asarray(value)
    expected_action_shape = (num_bs, num_prb)

    if actions_prb.shape != expected_action_shape:
        raise ValueError(
            f"PRB action shape must be {expected_action_shape}, got {actions_prb.shape}"
        )
    if actions_pwr.shape != expected_action_shape:
        raise ValueError(
            f"power action shape must be {expected_action_shape}, got {actions_pwr.shape}"
        )
    if log_probs.shape != (num_bs,):
        raise ValueError(f"log-probability shape must be ({num_bs},), got {log_probs.shape}")
    if value_array.ndim != 0:
        raise ValueError("centralized critic value must be scalar")
    if not (
        np.all(np.isfinite(actions_pwr))
        and np.all(np.isfinite(log_probs))
        and np.isfinite(value_array.item())
    ):
        raise ValueError("policy output contains NaN or Inf")

    return (
        actions_prb.astype(np.int64, copy=False),
        actions_pwr,
        log_probs,
        float(value_array.item()),
    )


def create_environment(config: dict[str, Any]) -> Any:
    """Create the real environment after all environment modules are integrated."""

    try:
        from ppo.env import MultiISACEnv
    except (ImportError, ModuleNotFoundError, SyntaxError) as exc:
        raise RuntimeError(
            "ppo.env or one of its physical-model dependencies is not ready. "
            "Integrate scenario/channel/sensing/communication before real training."
        ) from exc
    return MultiISACEnv(config)


def _serializable_info(info: Mapping[str, Any]) -> dict[str, float | bool | int]:
    scalars: dict[str, float | bool | int] = {}
    for key, value in info.items():
        if isinstance(value, (bool, np.bool_)):
            scalars[str(key)] = bool(value)
        elif isinstance(value, (int, np.integer)):
            scalars[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)) and np.isfinite(value):
            scalars[str(key)] = float(value)
    return scalars


def save_checkpoint(
    path: str | Path,
    agent: Any,
    config: Mapping[str, Any],
    episode: int,
    global_step: int,
) -> None:
    """Atomically save model, optimizer and random-generator state."""

    required = (
        "actornetwork",
        "criticnetwork",
        "actor_optimizer",
        "critic_optimizer",
    )
    missing = [name for name in required if not hasattr(agent, name)]
    if missing:
        raise TypeError(f"Agent cannot be checkpointed; missing: {', '.join(missing)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "episode": int(episode),
        "global_step": int(global_step),
        "config": dict(config),
        "actor": agent.actornetwork.state_dict(),
        "critic": agent.criticnetwork.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()

    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path: str | Path, agent: Any) -> tuple[int, int]:
    """Restore a checkpoint and return ``(next_episode, global_step)``."""

    checkpoint = torch.load(Path(path), map_location=agent.device, weights_only=False)
    agent.actornetwork.load_state_dict(checkpoint["actor"])
    agent.criticnetwork.load_state_dict(checkpoint["critic"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    return int(checkpoint["episode"]) + 1, int(checkpoint["global_step"])


def train(
    config: dict[str, Any],
    *,
    env: Any | None = None,
    agent: Any | None = None,
    buffer_factory: Callable[[], Any] | None = None,
    output_dir: str | Path = "results",
    resume_from: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run episodic on-policy MAPPO training and return episode metrics."""

    validate_config(config)
    system = config["system_parameters"]
    training = config["mappo_parameters"]["training"]
    num_bs = int(system["num_bs"])
    total_episodes = int(training["total_episodes"])
    episode_length = int(training["episode_length"])
    save_interval = int(training.get("save_interval", 0))
    log_interval = max(1, int(training.get("log_interval", 1)))
    seed = int(training.get("seed", 0))

    seed_everything(seed)
    env = env if env is not None else create_environment(config)
    initial_local_obs, initial_global_state = _reset_environment(env, seed)
    local_obs_dim, global_state_dim = infer_dimensions(
        initial_local_obs, initial_global_state, num_bs
    )

    if agent is None or buffer_factory is None:
        agent_type, buffer_type = _default_component_types()
        if agent is None:
            agent = agent_type(config, local_obs_dim, global_state_dim)
        if buffer_factory is None:
            buffer_factory = buffer_type

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_dir = output_dir / "checkpoints"

    start_episode = 0
    global_step = 0
    if resume_from is not None:
        start_episode, global_step = load_checkpoint(resume_from, agent)

    history: list[dict[str, Any]] = []
    file_mode = "a" if start_episode > 0 else "w"
    with metrics_path.open(file_mode, encoding="utf-8") as metrics_file:
        for episode in range(start_episode, total_episodes):
            local_obs, global_state = _reset_environment(env, seed + episode)
            infer_dimensions(local_obs, global_state, num_bs)
            buffer = buffer_factory()
            episode_return = 0.0
            final_info: dict[str, Any] = {}
            steps_taken = 0

            for step in range(episode_length):
                policy_output = agent.select_action(local_obs, global_state)
                actions_prb, actions_pwr, log_probs, value = _validate_policy_output(
                    *policy_output,
                    num_bs=num_bs,
                    num_prb=int(system["num_prb"]),
                )
                transition = _validate_step_result(
                    env.step(actions_prb, actions_pwr),
                    num_bs,
                    local_obs_dim,
                    global_state_dim,
                )
                (
                    next_local_obs,
                    next_global_state,
                    reward,
                    terminated,
                    env_truncated,
                    final_info,
                ) = transition

                reached_time_limit = step + 1 >= episode_length
                truncated = bool(env_truncated or reached_time_limit)
                buffer.push(
                    local_obs,
                    global_state,
                    actions_prb,
                    actions_pwr,
                    log_probs,
                    reward,
                    terminated,
                    truncated,
                    value,
                )

                local_obs = next_local_obs
                global_state = next_global_state
                episode_return += reward
                global_step += 1
                steps_taken += 1

                if terminated or truncated:
                    break

            if len(buffer) == 0:
                raise RuntimeError("No transitions were collected for the episode")

            update_metrics = agent.update(buffer, global_state)
            if not isinstance(update_metrics, Mapping):
                raise ValueError("agent.update() must return a metrics mapping")
            numeric_update_metrics = {
                str(key): float(value) for key, value in update_metrics.items()
            }
            if not all(np.isfinite(value) for value in numeric_update_metrics.values()):
                raise FloatingPointError("agent.update() returned NaN or Inf metrics")
            record: dict[str, Any] = {
                "episode": episode + 1,
                "global_step": global_step,
                "episode_return": float(episode_return),
                "episode_length": steps_taken,
                **_serializable_info(final_info),
                **numeric_update_metrics,
            }
            history.append(record)
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_file.flush()

            if (episode + 1) % log_interval == 0:
                print(
                    f"episode={episode + 1} steps={global_step} "
                    f"return={episode_return:.6f} length={steps_taken}"
                )

            if save_interval > 0 and (episode + 1) % save_interval == 0:
                save_checkpoint(
                    checkpoint_dir / f"episode_{episode + 1:06d}.pt",
                    agent,
                    config,
                    episode,
                    global_step,
                )

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-BS ISAC MAPPO agent")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config-dir", type=Path, default=project_root / "config")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_dir)
    train(config, output_dir=args.output_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
