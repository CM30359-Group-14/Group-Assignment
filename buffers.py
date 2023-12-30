import numpy as np

from collections import deque
from typing import Dict, Tuple


class ReplayBuffer:
    """
    Class representing a simple replay buffer that accepts stacked images.
    """
    
    def __init__(self, obs_shape: Tuple, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)

        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.max_size = size
        self.batch_size = batch_size

        self.ptr = 0
        self.size = 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        # Draw `batch_size` no. samples (without replacement) as indices.
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            indices=idxs,
        )
    
    def __len__(self) -> int:
        return self.size
    

class NStepReplayBuffer(ReplayBuffer):
    """
    Class representing a simple experience replay buffer for n-step learning.
    """

    def __init__(
        self,
        obs_shape: Tuple, 
        size: int, 
        batch_size: int = 32,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__(obs_shape, size, batch_size)

        # The n-step buffer stores at most `n_step` number of transitions.
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, next_obs: np.ndarray, done: bool) -> Tuple:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # If there aren't enough transitions for n-step learning yet.
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # Prepare an n-step transition.
        rew, next_obs, done = self._prepare_n_step_info()
        obs, act = self.n_step_buffer[0][:2]

        # Store the transition.
        super().store(obs, act, rew, next_obs, done)

        return self.n_step_buffer[0]
    
    def sample_batch_from_idxs(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )

    def _prepare_n_step_info(self) -> Tuple[np.float32, np.ndarray, bool]:
        """
        Returns the n_step reward, subsequent observation and whether or not
        the episode has completed.
        """
        rew, next_obs, done = self.n_step_buffer[-1][-3:]

        # Iterating backwards through the n-step buffer.
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self.gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done