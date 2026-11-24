"""Replay storage with copy isolation and checkpointable contents."""

from collections import deque
import copy
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *transition):
        self.buffer.append(tuple(np.array(value, copy=True) for value in transition))

    def sample(self, batch_size, device):
        batch = random.sample(list(self.buffer), batch_size)
        return tuple(
            torch.as_tensor(np.stack(column), dtype=torch.float32, device=device)
            for column in zip(*batch)
        )

    def state_dict(self):
        return {"capacity": self.capacity, "items": copy.deepcopy(list(self.buffer))}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or len(state["items"]) > self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        self.buffer = deque(copy.deepcopy(state["items"]), maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)
