import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from comUtils.common import get_activation, build_mlp

class Actor(nn.Module):
    """
    Decentralized Actor network with parameter sharing across base stations (BSs).
    Note:
        Operates under decentralized execution: each BS executes the policy using only its local observation.
    """
    def __init__(self, local_obs_dim, n_prb, num_targets, num_ues, net_conf, power_budget_control=False):
        """
        Initialize the Actor network modules and policy heads.

        Argument:
            local_obs_dim (int): Dimension of the local observation vector for each BS.
            n_prb (int): Number of Physical Resource Blocks (PRBs) per BS.
            num_targets (int): Number of mobile sensing targets.
            num_ues (int): Number of communication users (UEs).
            net_conf (dict): Network configuration
            power_budget_control (bool): Append one total-power fraction per BS.

        Return:
            None.
        Note:
            Optional group probabilities only initialize trainable PRB biases;
            they do not reserve PRBs or prevent later changes in allocation.
            The optional final power action controls the BS total budget.
        """
        super().__init__()
        self.n_prb = n_prb
        self.num_choices = 1 + num_targets + num_ues #number of choice for each prb
        self.power_action_dim = self.n_prb + int(power_budget_control)
        self.hiddens_dim = net_conf["hidden_dims"]
        # Use a common function to build the backbone
        self.backbone, self.out_gain = build_mlp(input_dim=local_obs_dim,
                                                hiddens_dim=net_conf["hidden_dims"],
                                                activation=net_conf["activation"],
                                                use_layer_norm=net_conf["use_layer_norm"],
                                                orthogonal_init=net_conf["orthogonal_init"],
                                                gain=net_conf["gain"],
                                                isActor=True)
        
        self.prb_head = nn.Linear(net_conf["hidden_dims"][1], self.n_prb * self.num_choices) # sum of prb choice
        self.power_mean = nn.Linear(net_conf["hidden_dims"][1], self.power_action_dim)
        if net_conf["orthogonal_init"]:
            for head in (self.prb_head, self.power_mean):
                nn.init.orthogonal_(head.weight, gain=self.out_gain)
                nn.init.zeros_(head.bias)
        # Khởi tạo logstd = -1.0 (std = exp(-1.0) ≈ 0.36) giúp độ tản ban đầu vừa vặn với miền [0, 1]
        self.power_logstd = nn.Parameter(torch.full((1, self.power_action_dim), -1.0))
        if "initial_sensing_fraction" in net_conf:
            sensing = float(net_conf["initial_sensing_fraction"])
            idle = float(net_conf.get("initial_idle_fraction", 0.02))
            if not (0 < sensing < 1 and 0 < idle < 1 and sensing + idle < 1):
                raise ValueError("Initial sensing/idle probabilities must be positive and sum to less than one")
            probabilities = [idle] + [sensing / num_targets] * num_targets
            probabilities += [(1 - sensing - idle) / num_ues] * num_ues
            with torch.no_grad():
                self.prb_head.bias.copy_(torch.tensor(
                    np.tile(np.log(probabilities), self.n_prb), dtype=self.prb_head.bias.dtype))
        if power_budget_control:
            fraction = float(net_conf.get("initial_power_fraction", 0.5))
            if not 0 < fraction < 1:
                raise ValueError("Initial power fraction must be strictly between zero and one")
            with torch.no_grad():
                self.power_mean.bias[-1] = np.log(fraction / (1 - fraction))
         
    def forward(self, obs, action_prb=None, action_power=None):
        """
        Forward pass for action sampling or evaluating log-probabilities and entropy.

        Argument:
            obs (torch.Tensor)[batch, local_obs_dim]: Local observation features of the base stations.
            action_prb (torch.Tensor or None)[batch, n_prb]: Optional discrete PRB actions for evaluation.
            action_power (torch.Tensor or None)[batch, power_action_dim]: PRB
                weights and, when enabled, one final BS power fraction.

        Return:
            action_prb (torch.Tensor)[batch, n_prb]: Sampled or evaluated discrete PRB selection indices.
            action_power (torch.Tensor)[batch, power_action_dim]: Normalized
                PRB weights and optional BS power fraction in [0, 1].
            total_log_prob (torch.Tensor)[batch]: Sum of log-probabilities for discrete and continuous action heads.
            total_entropy (torch.Tensor)[batch]: Total policy entropy of discrete and continuous distributions.

        Note:
            If actions are None, the method samples new actions. If provided, it evaluates their log-probabilities.
        """
        # 1. forward through backbone
        feat = self.backbone(obs)  # [batch, 128]

        # First branch: PRB (discrete)
        prb_logits = self.prb_head(feat).reshape(-1, self.n_prb, self.num_choices) 
        prb_dist = Categorical(logits=prb_logits)
        
        if action_prb is None:
            action_prb = prb_dist.sample() 
        
        # Calculate log_prob of PRB branch 
        logp_prb = prb_dist.log_prob(action_prb).sum(dim=-1)
        entropy_prb = prb_dist.entropy().sum(dim=-1)
    
        # Second branch: Power (continuous)
        raw_mean = self.power_mean(feat)
        std = torch.exp(self.power_logstd).expand_as(raw_mean)
        power_dist = Normal(raw_mean, std)

        if action_power is None:
            u = power_dist.sample()
            action_power = torch.sigmoid(u)
        else:
            # evaluation: clamp before logit to avoid NaN in the boundary
            clamped_action = action_power.clamp(1e-6, 1.0 - 1e-6)
            u = torch.log(clamped_action) - torch.log(1.0 - clamped_action)

        #softplus
        # ln(da/du) = -(softplus(u) + softplus(-u))
        log_jac = -(F.softplus(u) + F.softplus(-u))
        logp_power = (power_dist.log_prob(u) - log_jac).sum(dim=-1)
        entropy_power = power_dist.entropy().sum(dim=-1)

        # result
        total_log_prob = logp_prb + logp_power
        total_entropy = entropy_prb + entropy_power
        
        return action_prb, action_power, total_log_prob, total_entropy


class Critic(nn.Module):
    """
    Centralized Critic network for MAPPO.
    Estimates the expected global state-value V(S) for variance reduction during centralized training.

    Note:
        Operates under centralized training: accepts global state containing all BSs, targets, and channel info.
    """
    def __init__(self, global_state_dim, net_conf):
        """
        Initialize the Centralized Critic network backbone.

        Argument:
            global_state_dim (int): Dimension of the global state vector.
            net_conf (dict): Network configuration

        Return:
            None.
        """
        super().__init__()
        self.backbone, self.out_gain = build_mlp(input_dim=global_state_dim,
                        hiddens_dim=net_conf["hidden_dims"],
                        activation=net_conf["activation"],
                        use_layer_norm=net_conf["use_layer_norm"],
                        orthogonal_init=net_conf["orthogonal_init"],
                        gain=net_conf["gain"],
                        isActor=False)
        self.value_head = nn.Linear(net_conf["hidden_dims"][-1], 1)
        if net_conf["orthogonal_init"]:
            nn.init.orthogonal_(self.value_head.weight, gain=self.out_gain)
            nn.init.zeros_(self.value_head.bias)
    
    def forward(self, global_state):
        """
        Forward pass to compute state-value estimates.

        Argument:
            global_state (torch.Tensor)[batch, global_state_dim]: Global environment state vector.

        Return:
            value (torch.Tensor)[batch, 1]: Estimated scalar state-value V(S).
        """
        return self.value_head(self.backbone(global_state))


class MultiAgentRolloutBuffer:
    """
    Multi-agent experience replay rollout buffer.
    Stores trajectory rollout samples across T timesteps for B base stations.

    Note:
        Buffers are cleared after each policy update cycle.
    """
    def __init__(self):
        """
        Initialize empty transition storage lists for multi-agent rollouts.

        Argument:
            None.

        Return:
            None.
        """
        self.local_obs = []
        self.global_obs = []
        self.actions_prb = []
        self.actions_pwr = []
        self.log_probs = []
        self.rewards = []
        self.terminateds = []
        self.truncateds = []
        self.values = []
        self.next_values = []

    def push(self, local_obs, global_state, act_prb, act_pwr, logp, reward, terminated, truncated, value, next_value):
        """
        Append a single-step multi-agent transition into the buffer.

        Argument:
            local_obs (np.ndarray)[B, local_obs_dim]: Local observations for all B base stations.
            global_state (np.ndarray)[global_state_dim]: Global network state for Centralized Critic.
            act_prb (np.ndarray)[B, n_prb]: Discrete PRB selection decisions for all BSs.
            act_pwr (np.ndarray)[B, R or R+1]: PRB weights and optional BS budget.
            logp (np.ndarray)[B]: Joint log-probabilities of actions for each BS.
            reward (float): Shared team reward scalar received from the environment.
            terminated (bool or float): Terminal failure flag (e.g. lost target).
            truncated (bool or float): Timeout truncation flag (e.g. max steps reached).
            value (float): State-value V(S) predicted by the Centralized Critic.
            next_value (float): State-value of the state reached by this transition.

        Return:
            None.
        """
        self.local_obs.append(local_obs)
        self.global_obs.append(global_state)
        self.actions_prb.append(act_prb)
        self.actions_pwr.append(act_pwr)
        self.log_probs.append(logp)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.values.append(value)
        self.next_values.append(next_value)

    def clear(self):
        """
        Clear all stored rollout lists after a policy update.

        Argument:
            None.

        Return:
            None.
        """
        self.local_obs.clear()
        self.global_obs.clear()
        self.actions_prb.clear()
        self.actions_pwr.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.terminateds.clear()
        self.truncateds.clear()
        self.values.clear()
        self.next_values.clear()

    def get(self, device="cpu"):
        """
        Convert stored rollout lists into PyTorch tensors and transfer to target device.

        Argument:
            device (str or torch.device): Target device ('cpu' or 'cuda') for tensor storage.

        Return:
            b_obs (torch.Tensor)[T, B, local_obs_dim]: Batched local observations.
            b_states (torch.Tensor)[T, global_state_dim]: Batched global state vectors.
            b_act_prb (torch.Tensor)[T, B, n_prb]: Batched discrete PRB decisions (dtype int64).
            b_act_pwr (torch.Tensor)[T, B, R or R+1]: Batched power actions.
            b_logprobs (torch.Tensor)[T, B]: Batched action log-probabilities.
            b_rewards (torch.Tensor)[T]: Batched team rewards.
            b_terminateds (torch.Tensor)[T]: Batched terminal state flags.
            b_truncateds (torch.Tensor)[T]: Batched timeout truncation flags.
            b_values (torch.Tensor)[T]: Batched Centralized Critic value estimates.
        """
        # Re-optimize to avoid twice copy (list -> numpy -> tensor)
        b_obs = torch.as_tensor(np.array(self.local_obs), dtype=torch.float32, device=device)
        b_states = torch.as_tensor(np.array(self.global_obs), dtype=torch.float32, device=device)
        b_act_prb = torch.as_tensor(np.array(self.actions_prb), dtype=torch.int64, device=device)
        b_act_pwr = torch.as_tensor(np.array(self.actions_pwr), dtype=torch.float32, device=device)
        b_logprobs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        b_rewards = torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        # terminated and truncated should use torch.bool
        b_terminateds = torch.as_tensor(np.array(self.terminateds), dtype=torch.bool, device=device)
        b_truncateds = torch.as_tensor(np.array(self.truncateds), dtype=torch.bool, device=device)
        b_values = torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device)

        return b_obs, b_states, b_act_prb, b_act_pwr, b_logprobs, b_rewards, b_terminateds, b_truncateds, b_values

    def __len__(self):
        """
        Get the current number of timesteps stored in the buffer.

        Argument:
            None.

        Return:
            length (int): Number of rollout steps T currently stored.
        """
        return len(self.local_obs)
