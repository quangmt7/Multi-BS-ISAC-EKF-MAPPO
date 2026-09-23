import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class Actor(nn.Module):
    """
    Decentralized Actor network with parameter sharing across base stations (BSs).
    Features a dual-head structure for hybrid action spaces: discrete PRB assignment and continuous power allocation.

    Note:
        Operates under decentralized execution: each BS executes the policy using only its local observation.
    """
    def __init__(self, local_obs_dim, n_prb, num_targets, num_ues, h_dim=128):
        """
        Initialize the Actor network modules and policy heads.

        Argument:
            local_obs_dim (int): Dimension of the local observation vector for each BS.
            n_prb (int): Number of Physical Resource Blocks (PRBs) per BS.
            num_targets (int): Number of mobile sensing targets.
            num_ues (int): Number of communication users (UEs).
            h_dim (int): Hidden dimension for MLP backbone layers. Default: 128.

        Return:
            None.
        """
        super().__init__()
        self.n_prb = n_prb
        self.num_choices = 1 + num_targets + num_ues #number of choice for each prb 
        
        self.backbone= nn.Sequential(
            nn.Linear(local_obs_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh()
        )
        
        self.prb_head =nn.Linear(h_dim, self.n_prb*self.num_choices) # sum of prb choice 
        self.power_mean= nn.Linear(h_dim, self.n_prb) # n numbers of power allocation for n prb
        self.power_logstd = nn.Parameter(torch.zeros(1, self.n_prb))
         
    def forward(self, obs, action_prb=None, action_power=None):
        """
        Forward pass for action sampling or evaluating log-probabilities and entropy.

        Argument:
            obs (torch.Tensor)[batch, local_obs_dim]: Local observation features of the base stations.
            action_prb (torch.Tensor or None)[batch, n_prb]: Optional discrete PRB actions for evaluation.
            action_power (torch.Tensor or None)[batch, n_prb]: Optional continuous power actions for evaluation.

        Return:
            action_prb (torch.Tensor)[batch, n_prb]: Sampled or evaluated discrete PRB selection indices.
            action_power (torch.Tensor)[batch, n_prb]: Sampled or evaluated continuous normalized power values in [0, 1].
            total_log_prob (torch.Tensor)[batch]: Sum of log-probabilities for discrete and continuous action heads.
            total_entropy (torch.Tensor)[batch]: Total policy entropy of discrete and continuous distributions.

        Note:
            If actions are None, the method samples new actions. If provided, it evaluates their log-probabilities.
        """
        # 1. foward through backbone
        feat = self.backbone(obs)  # [batch, 128]
        # First branch: PRB (discrete)
        prb_logits = self.prb_head(feat).reshape(-1, self.n_prb, self.num_choices) 
        # Categorical 
        prb_dist = Categorical(logits=prb_logits)
        
        if action_prb is None:
            action_prb = prb_dist.sample() 
        
        # Calculate log_prob of PRB branch 
        logp_prb = prb_dist.log_prob(action_prb).sum(dim=-1)
        entropy_prb = prb_dist.entropy().sum(dim=-1)
    
        # Second branch: Power ( continuous)
        # Sigmoid function
        mean = torch.sigmoid(self.power_mean(feat))
        std = torch.exp(self.power_logstd).expand_as(mean)
        power_dist = Normal(mean, std)
        if action_power is None:
            action_power = power_dist.sample()  
            
        # calculate log_prob of power branch
        logp_power = power_dist.log_prob(action_power).sum(dim=-1)
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
    def __init__(self, global_state_dim, h_dim=256):
        """
        Initialize the Centralized Critic network backbone.

        Argument:
            global_state_dim (int): Dimension of the global state vector.
            h_dim (int): Hidden dimension for MLP layers. Default: 256.

        Return:
            None.
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(global_state_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1)
        )
    
    def forward(self, global_state):
        """
        Forward pass to compute state-value estimates.

        Argument:
            global_state (torch.Tensor)[batch, global_state_dim]: Global environment state vector.

        Return:
            value (torch.Tensor)[batch, 1]: Estimated scalar state-value V(S).
        """
        return self.backbone(global_state)


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
        self.dones = []
        self.values = []

    def push(self, local_obs, global_state, act_prb, act_pwr, logp, reward, done, value):
        """
        Append a single-step multi-agent transition into the buffer.

        Argument:
            local_obs (np.ndarray)[B, local_obs_dim]: Local observations for all B base stations.
            global_state (np.ndarray)[global_state_dim]: Global network state for Centralized Critic.
            act_prb (np.ndarray)[B, n_prb]: Discrete PRB selection decisions for all BSs.
            act_pwr (np.ndarray)[B, n_prb]: Continuous power allocation decisions for all BSs.
            logp (np.ndarray)[B]: Joint log-probabilities of actions for each BS.
            reward (float): Shared team reward scalar received from the environment.
            done (bool or float): Episode termination flag.
            value (float): State-value V(S) predicted by the Centralized Critic.

        Return:
            None.
        """
        self.local_obs.append(local_obs)
        self.global_obs.append(global_state)
        self.actions_prb.append(act_prb)
        self.actions_pwr.append(act_pwr)
        self.log_probs.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

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
        self.dones.clear()
        self.values.clear()

    def get(self, device="cpu"):
        """
        Convert stored rollout lists into PyTorch tensors and transfer to target device.

        Argument:
            device (str or torch.device): Target device ('cpu' or 'cuda') for tensor storage.

        Return:
            b_obs (torch.Tensor)[T, B, local_obs_dim]: Batched local observations.
            b_states (torch.Tensor)[T, global_state_dim]: Batched global state vectors.
            b_act_prb (torch.Tensor)[T, B, n_prb]: Batched discrete PRB decisions (dtype int64).
            b_act_pwr (torch.Tensor)[T, B, n_prb]: Batched continuous power allocations (dtype float32).
            b_logprobs (torch.Tensor)[T, B]: Batched action log-probabilities.
            b_rewards (torch.Tensor)[T]: Batched team rewards.
            b_dones (torch.Tensor)[T]: Batched termination flags.
            b_values (torch.Tensor)[T]: Batched Centralized Critic value estimates.
        """
        b_obs = torch.tensor(np.array(self.local_obs), dtype=torch.float32, device=device)
        b_states = torch.tensor(np.array(self.global_obs), dtype=torch.float32, device=device)
        b_act_prb = torch.tensor(np.array(self.actions_prb), dtype=torch.int64, device=device)
        b_act_pwr = torch.tensor(np.array(self.actions_pwr), dtype=torch.float32, device=device)
        b_logprobs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        b_rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        b_dones = torch.tensor(np.array(self.dones), dtype=torch.float32, device=device)
        b_values = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)

        return b_obs, b_states, b_act_prb, b_act_pwr, b_logprobs, b_rewards, b_dones, b_values

    def __len__(self):
        """
        Get the current number of timesteps stored in the buffer.

        Argument:
            None.

        Return:
            length (int): Number of rollout steps T currently stored.
        """
        return len(self.local_obs)


class MAPPOAgent:
    """
    Multi-Agent Proximal Policy Optimization controller.
    Manages decentralized Actor policies with Parameter Sharing and a Centralized Critic for cooperative ISAC.
    """
    def __init__(
        self,
        local_obs_dim,
        global_state_dim,
        n_prb,
        num_targets,
        num_ues,
        num_bs,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        batch_size=64,
        device=None
    ):
        """
        Initialize networks, dual optimizers, and algorithmic hyperparameters for MAPPO.

        Argument:
            local_obs_dim (int): Dimension of the local observation vector for each BS.
            global_state_dim (int): Dimension of the global state vector.
            n_prb (int): Number of PRBs per BS.
            num_targets (int): Number of mobile sensing targets.
            num_ues (int): Number of communication users (UEs).
            num_bs (int): Number of base stations (agents).
            lr_actor (float): Learning rate for the Actor optimizer. Default: 3e-4.
            lr_critic (float): Learning rate for the Critic optimizer. Default: 3e-4.
            gamma (float): Discount factor for future rewards. Default: 0.99.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation. Default: 0.95.
            clip_coef (float): PPO surrogate clipping threshold epsilon. Default: 0.2.
            ent_coef (float): Entropy bonus coefficient. Default: 0.01.
            vf_coef (float): Value loss coefficient. Default: 0.5.
            max_grad_norm (float): Maximum norm for gradient clipping. Default: 0.5.
            update_epochs (int): Number of optimization passes over rollout data. Default: 10.
            batch_size (int): Minibatch size for gradient updates. Default: 64.
            device (str, torch.device, or None): Device for tensor computations ('cpu', 'cuda', or None for auto).

        Return:
            None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        # Khởi tạo Actor, Critic, Optimizers và lưu hyperparameter
        self.actornetwork = Actor(local_obs_dim, n_prb, num_targets, num_ues).to(self.device)
        self.criticnetwork = Critic(global_state_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actornetwork.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.criticnetwork.parameters(), lr=lr_critic, eps=1e-5)


        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

    def select_action(self, local_obs_list, global_state):
        """
        Select actions for all B base stations and evaluate global state-value during environment interaction.

        Argument:
            local_obs_list (np.ndarray)[B, local_obs_dim]: Local observations for all B base stations.
            global_state (np.ndarray)[global_state_dim]: Global network state vector.

        Return:
            act_prb (np.ndarray)[B, n_prb]: Discrete PRB selection decisions for B BSs (dtype int64).
            act_pwr (np.ndarray)[B, n_prb]: Continuous normalized power allocations for B BSs in [0, 1] (dtype float32).
            logp (np.ndarray)[B]: Action log-probabilities for each BS.
            val (float): Scalar state-value V(S) predicted by the Centralized Critic.

        Note:
            Decentralized execution: Actor processes B agents concurrently via parameter sharing.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(local_obs_list, dtype=torch.float32, device=self.device)
            state_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
            if state_t.ndim ==1:
                state_t=state_t.unsqueeze(0)
                
            act_prb, act_pwr, logp, _ = self.actornetwork(obs_t)
            val = self.criticnetwork(state_t)
            return (
                act_prb.cpu().numpy(),
                act_pwr.cpu().numpy(),
                logp.cpu().numpy(),
                val.squeeze().item()
            )

    def compute_gae(self, buffer, next_global_state, next_done):
        """
        Compute Generalized Advantage Estimation (GAE) and discounted return targets backwards across rollout steps.

        Argument:
            buffer (MultiAgentRolloutBuffer): Rollout buffer holding T timesteps of multi-agent interactions.
            next_global_state (np.ndarray)[global_state_dim]: Global state of the timestep following the rollout (S_T).
            next_done (bool or float): Done flag for the final transition.

        Return:
            advantages_t (torch.Tensor)[T]: Computed Generalized Advantage Estimations.
            returns_t (torch.Tensor)[T]: Discounted target returns for Critic training.

        Note:
            Bootstrap value is computed from Centralized Critic on next_global_state.
        """
        with torch.no_grad():
            next_state_t = torch.as_tensor(next_global_state, dtype=torch.float32, device=self.device)
            if next_state_t.ndim == 1:
                next_state_t = next_state_t.unsqueeze(0)

            next_val = self.criticnetwork(next_state_t).squeeze().item()
            rewards = buffer.rewards
            dones = buffer.dones
            values = buffer.values
            T = len(rewards)

            advantages = [0.0] * T
            last_gaelam = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_val_step = next_val
                    next_nonterminal = 1.0 - float(next_done)
                else:
                    next_val_step = values[t + 1]
                    next_nonterminal = 1.0 - float(dones[t])

                # TD Error: delta = r + gamma * V(s') - V(s)
                delta = rewards[t] + self.gamma * next_val_step * next_nonterminal - values[t]

                # GAE: A_t = delta + gamma * lambda * next_nonterminal * A_{t+1}
                advantages[t] = last_gaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gaelam

            returns = [adv + val for adv, val in zip(advantages, values)]
            # convert and return Advantages, Q 
            advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

            return advantages_t, returns_t
        
    def update(self, buffer, next_global_state, next_done):
        """
        Update Actor and Centralized Critic networks using PPO clipped loss over multiple epochs.

        Argument:
            buffer (MultiAgentRolloutBuffer): Buffer containing on-policy rollout transitions.
            next_global_state (np.ndarray)[global_state_dim]: Global state after rollout horizon.
            next_done (bool or float): Done flag for the final transition.

        Return:
            metrics (dict): Dictionary containing average training losses:
                - 'policy_loss' (float): Average PPO clipped policy loss.
                - 'critic_loss' (float): Average mean-squared error value loss.
                - 'entropy' (float): Average policy distribution entropy.

        Note:
            Parameter Sharing flattens Actor samples into (T * B), whereas Centralized Critic trains on T global states.
        """
        
        b_advantages, b_returns =self.compute_gae(buffer,next_global_state, next_done)
        b_obs, b_states, b_act_prb, b_act_pwr, b_log_probs, _ , _ ,b_values = buffer.get(self.device)        
        #normalization
        b_advantages = (b_advantages -b_advantages.mean())/ (b_advantages.std() + 1e-8)
        
        #Flatten data : flat T and B 
        T,B =b_obs.shape[0], b_obs.shape[1]
        # [T,B,...] --> [T*B,...]
        flat_obs = b_obs.reshape(-1, b_obs.shape[-1])
        flat_act_prb = b_act_prb.reshape(-1, b_act_prb.shape[-1])
        flat_act_pwr = b_act_pwr.reshape(-1, b_act_pwr.shape[-1])
        flat_logp = b_log_probs.reshape(-1)
        # 
        flat_adv = b_advantages.unsqueeze(1).repeat(1, B).reshape(-1)     
        
        
        
        # define statistical array
        pg_losses = []   # Policy Gradient Loss (L_clip)
        v_losses = []    # Value Loss (L_VF)
        ent_losses = []  # Entropy Loss
        
        total_actor_samples = T*B
        actor_indices= np.arange(total_actor_samples)
        critic_indices=np.arange(T)
        
        # epochs loop
        for epoch in range(self.update_epochs):
            np.random.shuffle(actor_indices)
            np.random.shuffle(critic_indices)

            # UPDATE ACTOR  
            for start in range(0, total_actor_samples, self.batch_size):
                end = start + self.batch_size
                mb_inds = actor_indices[start:end]

                mb_obs = flat_obs[mb_inds]
                mb_act_prb = flat_act_prb[mb_inds]
                mb_act_pwr = flat_act_pwr[mb_inds]
                mb_logp = flat_logp[mb_inds]
                mb_advantages = flat_adv[mb_inds]

                # add to actor network 
                _, _, new_logp, entropy = self.actornetwork(mb_obs, mb_act_prb, mb_act_pwr)
                
                # calculate prob ratio
                ratio = torch.exp(new_logp - mb_logp)

                # ppo CLIP
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus
                entropy_loss = -entropy.mean()
                # total actor loss
                actor_loss = policy_loss + self.ent_coef * entropy_loss

                # Backward and optim
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actornetwork.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                pg_losses.append(policy_loss.item())
                ent_losses.append(entropy.mean().item())

            # UPDATE CRITIC
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                mb_inds = critic_indices[start:end]

                mb_states = b_states[mb_inds]
                mb_returns = b_returns[mb_inds]
                new_val = self.criticnetwork(mb_states).squeeze(-1)

                # cal MSE loss
                critic_loss = 0.5 * ((new_val - mb_returns) ** 2).mean()

                # backward and optim
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.criticnetwork.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                v_losses.append(critic_loss.item())
        # CLEAR BUFFER AND RETURN LOG
        buffer.clear()

        return {
            "policy_loss": np.mean(pg_losses),
            "critic_loss": np.mean(v_losses),
            "entropy": np.mean(ent_losses)
        }