"""Parameter-sharing MAPPO with a PRB-only masked policy."""

import numpy as np
import torch
from torch import nn
from ppo.agents.AgentComponents import Actor, Critic


class MAPPOAgent:
    def __init__(self, config, local_obs_dim, global_state_dim, *, single_bwp=False):
        self.config = config
        self.single_bwp = bool(single_bwp)
        cfg = config["mappo_parameters"]
        self.ppo = cfg["ppo"]
        device = cfg["training"]["device"]
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.actornetwork = Actor(local_obs_dim, config).to(self.device)
        self.criticnetwork = Critic(global_state_dim, cfg["network"]["critic"]).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actornetwork.parameters(), lr=self.ppo["lr_actor"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.criticnetwork.parameters(), lr=self.ppo["lr_critic"]
        )

    def select_action(self, local_obs_list, global_state, *, deterministic=False):
        with torch.no_grad():
            obs = torch.as_tensor(
                local_obs_list, dtype=torch.float32, device=self.device
            )
            action, logp, _ = self.actornetwork(
                obs, single_bwp=self.single_bwp, deterministic=deterministic
            )
            value = self.estimate_value(global_state)
        return action.cpu().numpy(), logp.cpu().numpy(), value

    def estimate_value(self, global_state):
        with torch.no_grad():
            state = torch.as_tensor(
                global_state, dtype=torch.float32, device=self.device
            )
            return float(self.criticnetwork(state.unsqueeze(0)).item())

    def compute_gae(self, buffer):
        data = buffer.get(self.device)
        rewards, terminated, truncated, values, next_values = data[4:]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=self.device)
        gamma, lam = self.ppo["gamma"], self.ppo["gae_lambda"]
        for t in reversed(range(len(buffer))):
            bootstrap = (~terminated[t]).float()
            continue_trace = (~(terminated[t] | truncated[t])).float()
            delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
            gae = delta + gamma * lam * continue_trace * gae
            advantages[t] = gae
        return advantages, advantages + values

    def update_learning_rate(self, current_episode, decay_episodes):
        fraction = max(0.2, 1.0 - (current_episode + 1) / decay_episodes)
        for optimizer, key in (
            (self.actor_optimizer, "lr_actor"),
            (self.critic_optimizer, "lr_critic"),
        ):
            for group in optimizer.param_groups:
                group["lr"] = fraction * self.ppo[key]

    def update(self, buffer):
        advantages, returns = self.compute_gae(buffer)
        obs, states, actions, old_logp, _, _, _, _, _ = buffer.get(self.device)
        t, b = obs.shape[:2]
        obs, actions, old_logp = (
            obs.reshape(t * b, -1),
            actions.reshape(t * b, -1),
            old_logp.reshape(-1),
        )
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp_min(1e-8)
        advantages = advantages[:, None].expand(t, b).reshape(-1)
        metrics = {
            k: []
            for k in (
                "policy_loss",
                "critic_loss",
                "entropy",
                "ratio_mean",
                "ratio_std",
                "clip_fraction",
            )
        }
        batch_size = self.ppo["batch_size"]
        for _ in range(self.ppo["ppo_epochs"]):
            for indices in torch.randperm(t * b, device=self.device).split(batch_size):
                _, logp, entropy = self.actornetwork(
                    obs[indices], actions[indices], single_bwp=self.single_bwp
                )
                ratio = (logp - old_logp[indices]).exp()
                surrogate = torch.minimum(
                    ratio * advantages[indices],
                    ratio.clamp(
                        1 - self.ppo["clip_epsilon"], 1 + self.ppo["clip_epsilon"]
                    )
                    * advantages[indices],
                )
                policy_loss = -surrogate.mean()
                loss = policy_loss - self.ppo["entropy_coef"] * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actornetwork.parameters(),
                    self.ppo["max_grad_norm"],
                    error_if_nonfinite=True,
                )
                self.actor_optimizer.step()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["ratio_mean"].append(ratio.mean().item())
                metrics["ratio_std"].append(ratio.std(unbiased=False).item())
                metrics["clip_fraction"].append(
                    ((ratio - 1).abs() > self.ppo["clip_epsilon"]).float().mean().item()
                )
            for indices in torch.randperm(t, device=self.device).split(batch_size):
                predictions = self.criticnetwork(states[indices]).squeeze(-1)
                errors = (predictions - returns[indices]).square()
                value_loss = 0.5 * errors.mean() * self.ppo["value_loss_coef"]
                if not torch.isfinite(value_loss):
                    raise FloatingPointError("Non-finite PPO critic loss")
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.criticnetwork.parameters(),
                    self.ppo["max_grad_norm"],
                    error_if_nonfinite=True,
                )
                self.critic_optimizer.step()
                metrics["critic_loss"].append(value_loss.item())
        buffer.clear()
        return {name: float(np.mean(values)) for name, values in metrics.items()}
