import json

import numpy as np
import pytest
import torch

from ppo.train import (
    infer_dimensions,
    load_checkpoint,
    load_config,
    save_checkpoint,
    train,
)


class FakeEnv:
    def __init__(self, num_bs=2, obs_dim=3, episode_length=3):
        self.num_bs = num_bs
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        local_obs = np.zeros((self.num_bs, self.obs_dim), dtype=np.float32)
        global_state = local_obs.reshape(-1)
        return local_obs, global_state

    def step(self, actions_prb, actions_pwr):
        self.step_count += 1
        local_obs = np.full(
            (self.num_bs, self.obs_dim), self.step_count / self.episode_length,
            dtype=np.float32,
        )
        global_state = local_obs.reshape(-1)
        terminated = False
        truncated = self.step_count >= self.episode_length
        info = {"aoi_mean": float(self.step_count), "sum_rate_mbps": 2.5}
        return local_obs, global_state, 1.0, terminated, truncated, info


class FakeBuffer:
    def __init__(self):
        self.rows = []

    def push(self, *row):
        self.rows.append(row)

    def clear(self):
        self.rows.clear()

    def __len__(self):
        return len(self.rows)


class FakeAgent:
    def __init__(self, num_bs=2, n_prb=4):
        self.num_bs = num_bs
        self.n_prb = n_prb
        self.update_lengths = []

    def select_action(self, local_obs, global_state):
        actions_prb = np.zeros((self.num_bs, self.n_prb), dtype=np.int64)
        actions_pwr = np.zeros((self.num_bs, self.n_prb), dtype=np.float32)
        log_probs = np.zeros(self.num_bs, dtype=np.float32)
        return actions_prb, actions_pwr, log_probs, 0.0

    def update(self, buffer, next_global_state):
        self.update_lengths.append(len(buffer))
        buffer.clear()
        return {"policy_loss": 0.1, "critic_loss": 0.2, "entropy": 0.3}


def minimal_config():
    return {
        "system_parameters": {
            "num_bs": 2,
            "num_ue": 1,
            "num_target": 1,
            "num_prb": 4,
            "simulation": {"num_slots_per_episode": 3},
        },
        "ekf_parameters": {"state_dim": 6},
        "mappo_parameters": {
            "training": {
                "total_episodes": 2,
                "episode_length": 3,
                "save_interval": 0,
                "log_interval": 1,
                "seed": 7,
            }
        },
    }


def test_load_config_merges_the_three_owned_sections(tmp_path):
    (tmp_path / "system.yaml").write_text(
        "system_parameters:\n  num_bs: 2\n  num_ue: 1\n  num_target: 1\n  num_prb: 4\n",
        encoding="utf-8",
    )
    (tmp_path / "ekf.yaml").write_text(
        "ekf_parameters:\n  state_dim: 6\n", encoding="utf-8"
    )
    (tmp_path / "mappo.yaml").write_text(
        "mappo_parameters:\n  training:\n    total_episodes: 2\n    episode_length: 3\n",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config["system_parameters"]["num_bs"] == 2
    assert config["ekf_parameters"]["state_dim"] == 6
    assert config["mappo_parameters"]["training"]["total_episodes"] == 2


def test_infer_dimensions_rejects_wrong_number_of_bs():
    with pytest.raises(ValueError, match="num_bs"):
        infer_dimensions(np.zeros((3, 4)), np.zeros(12), num_bs=2)


def test_config_rejects_clipped_value_loss_for_mse_baseline(tmp_path):
    config = minimal_config()
    config["mappo_parameters"]["ppo"] = {"use_clipped_value_loss": True}

    from ppo.train import validate_config

    with pytest.raises(ValueError, match="without clipped value loss"):
        validate_config(config)


def test_train_collects_full_episodes_and_writes_jsonl(tmp_path):
    agent = FakeAgent()

    history = train(
        minimal_config(),
        env=FakeEnv(),
        agent=agent,
        buffer_factory=FakeBuffer,
        output_dir=tmp_path,
    )

    assert agent.update_lengths == [3, 3]
    assert [row["episode_return"] for row in history] == [3.0, 3.0]
    assert [row["episode_length"] for row in history] == [3, 3]
    assert history[-1]["aoi_mean"] == 3.0
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[-1]["critic_loss"] == 0.2


def test_train_rejects_non_scalar_team_reward(tmp_path):
    class InvalidRewardEnv(FakeEnv):
        def step(self, actions_prb, actions_pwr):
            values = list(super().step(actions_prb, actions_pwr))
            values[2] = np.array([1.0, 1.0])
            return tuple(values)

    with pytest.raises(ValueError, match="team reward"):
        train(
            minimal_config(),
            env=InvalidRewardEnv(),
            agent=FakeAgent(),
            buffer_factory=FakeBuffer,
            output_dir=tmp_path,
        )


def test_train_rejects_policy_action_with_wrong_shape(tmp_path):
    class InvalidActionAgent(FakeAgent):
        def select_action(self, local_obs, global_state):
            actions_prb, actions_pwr, log_probs, value = super().select_action(
                local_obs, global_state
            )
            return actions_prb[:, :-1], actions_pwr, log_probs, value

    with pytest.raises(ValueError, match="PRB action shape"):
        train(
            minimal_config(),
            env=FakeEnv(),
            agent=InvalidActionAgent(),
            buffer_factory=FakeBuffer,
            output_dir=tmp_path,
        )


def test_checkpoint_restores_network_and_optimizer_state(tmp_path):
    class CheckpointAgent:
        def __init__(self):
            self.device = torch.device("cpu")
            self.actornetwork = torch.nn.Linear(2, 1)
            self.criticnetwork = torch.nn.Linear(2, 1)
            self.actor_optimizer = torch.optim.Adam(
                self.actornetwork.parameters(), lr=1e-3
            )
            self.critic_optimizer = torch.optim.Adam(
                self.criticnetwork.parameters(), lr=2e-3
            )

    agent = CheckpointAgent()
    original_weight = agent.actornetwork.weight.detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, agent, minimal_config(), episode=4, global_step=15)

    with torch.no_grad():
        agent.actornetwork.weight.add_(10.0)
    next_episode, global_step = load_checkpoint(path, agent)

    assert next_episode == 5
    assert global_step == 15
    assert torch.equal(agent.actornetwork.weight, original_weight)
