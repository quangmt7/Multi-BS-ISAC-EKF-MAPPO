"""SAC in raw tanh action coordinates; Env ratios are a deterministic decoder."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from comUtils.common import build_mlp
from ppo.utils.project_config import hier_hidden_dims
from ppo.agents.replay import ReplayBuffer


class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.backbone, _ = build_mlp(state_dim, hidden, "relu", True, True, 1.0, True)
        self.mean = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Linear(hidden[-1], action_dim)
        for layer in (self.mean, self.log_std):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def sample(self, state):
        features = self.backbone(state)
        mean = self.mean(features)
        std = self.log_std(features).clamp(-5.0, 2.0).exp()
        distribution = Normal(mean, std)
        latent = distribution.rsample()
        action = latent.tanh()
        # Stable log-Jacobian of tanh, including saturated samples.
        log_jacobian = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
        log_prob = (distribution.log_prob(latent) - log_jacobian).sum(
            dim=-1, keepdim=True
        )
        return action, log_prob, mean.tanh()

    def forward(self, state):
        return self.mean(self.backbone(state)).tanh()


class _SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.backbone, gain = build_mlp(
            state_dim + action_dim, hidden, "relu", True, True, 1.0, False
        )
        self.head = nn.Linear(hidden[-1], 1)
        nn.init.orthogonal_(self.head.weight, gain=gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, state, action):
        return self.head(self.backbone(torch.cat((state, action), dim=-1)))


class SACAgent:
    def __init__(self, config, state_dim, device="cpu"):
        cfg = config["hierarchical_parameters"]["sac"]
        system = config["system_parameters"]
        self.device = torch.device(device)
        self.action_shape = (
            system["num_bs"],
            len(system["nPrb"]),
            system["num_target"] + system["num_embb"] + system["num_urllc"],
        )
        self.action_dim = int(np.prod(self.action_shape))
        hidden = hier_hidden_dims(cfg)
        self.actor = GaussianActor(state_dim, self.action_dim, hidden).to(self.device)
        self.critic_1 = _SACCritic(state_dim, self.action_dim, hidden).to(self.device)
        self.critic_2 = _SACCritic(state_dim, self.action_dim, hidden).to(self.device)
        self.target_critic_1 = _SACCritic(state_dim, self.action_dim, hidden).to(
            self.device
        )
        self.target_critic_2 = _SACCritic(state_dim, self.action_dim, hidden).to(
            self.device
        )
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_1_opt = torch.optim.Adam(
            self.critic_1.parameters(), lr=cfg["critic_lr"]
        )
        self.critic_2_opt = torch.optim.Adam(
            self.critic_2.parameters(), lr=cfg["critic_lr"]
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(cfg["init_alpha"]), device=self.device)
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["alpha_lr"])
        self.gamma, self.tau = cfg["gamma"], cfg["tau"]
        self.batch_size = cfg["batch_size"]
        self.policy_delay = cfg["policy_delay"]
        self.grad_clip = cfg.get("grad_clip", cfg.get("max_grad_norm", 5.0))
        self.auto_entropy = cfg.get("auto_entropy", True)
        self.update_after = max(
            self.batch_size,
            cfg.get("warmup_steps", cfg.get("update_after", self.batch_size)),
        )
        entropy = cfg.get("target_entropy")
        self.target_entropy = (
            -0.5 * self.action_dim
            if entropy is None or entropy == "auto"
            else float(entropy)
        )
        self.replay_buffer = ReplayBuffer(cfg["replay_capacity"])
        self.update_step = 0

    @property
    def alpha(self):
        return float(self.log_alpha.detach().exp().item())

    def select_action(self, state, *, deterministic=False):
        with torch.no_grad():
            state = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = self.actor(state) if deterministic else self.actor.sample(state)[0]
        return action.squeeze(0).cpu().numpy()

    def update(self):
        metrics = {
            "sac_actor_loss": 0.0,
            "sac_critic_loss": 0.0,
            "sac_alpha": self.alpha,
            "updated": False,
            "actor_updated": False,
        }
        if len(self.replay_buffer) < self.update_after:
            return metrics
        state, action, reward, next_state, terminated = self.replay_buffer.sample(
            self.batch_size, self.device
        )
        reward, terminated = reward.reshape(-1, 1), terminated.reshape(-1, 1)
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_state)
            next_q = torch.minimum(
                self.target_critic_1(next_state, next_action),
                self.target_critic_2(next_state, next_action),
            )
            target = reward + self.gamma * (1.0 - terminated) * (
                next_q - self.alpha * next_logp
            )
        loss_1 = F.smooth_l1_loss(self.critic_1(state, action), target)
        loss_2 = F.smooth_l1_loss(self.critic_2(state, action), target)
        critic_loss = loss_1 + loss_2
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite SAC critic loss")
        self.critic_1_opt.zero_grad(set_to_none=True)
        self.critic_2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        for network in (self.critic_1, self.critic_2):
            nn.utils.clip_grad_norm_(
                network.parameters(), self.grad_clip, error_if_nonfinite=True
            )
        self.critic_1_opt.step()
        self.critic_2_opt.step()
        self.update_step += 1
        if self.update_step % self.policy_delay == 0:
            for network in (self.critic_1, self.critic_2):
                network.requires_grad_(False)
            try:
                sampled, logp, _ = self.actor.sample(state)
                q = torch.minimum(
                    self.critic_1(state, sampled), self.critic_2(state, sampled)
                )
                actor_loss = (self.log_alpha.exp().detach() * logp - q).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.grad_clip, error_if_nonfinite=True
                )
                self.actor_opt.step()
            finally:
                for network in (self.critic_1, self.critic_2):
                    network.requires_grad_(True)
            if self.auto_entropy:
                alpha_loss = -(
                    self.log_alpha * (logp.detach() + self.target_entropy)
                ).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
            metrics["actor_updated"] = True
            metrics["sac_actor_loss"] = float(actor_loss.item())
        with torch.no_grad():
            for source, target_net in (
                (self.critic_1, self.target_critic_1),
                (self.critic_2, self.target_critic_2),
            ):
                for parameter, target_parameter in zip(
                    source.parameters(), target_net.parameters()
                ):
                    target_parameter.lerp_(parameter, self.tau)
        metrics.update(
            sac_critic_loss=float(critic_loss.item()),
            sac_alpha=self.alpha,
            updated=True,
        )
        return metrics
