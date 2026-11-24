from .env import BaseISACEnv


class PPOISACEnv(BaseISACEnv):
    def _get_global_state(self):
        """Centralized critic observes all per-BS DQN/PPO features."""
        return self._get_local_observations().ravel()
