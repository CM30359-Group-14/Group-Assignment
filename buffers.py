import random
import numpy as np

from collections import deque
from typing import Dict, Tuple, List

from trees import SumSegmentTree, MinSegmentTree


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
    

class PrioritisedReplayBuffer(NStepReplayBuffer):
    """
    Class representing a Prioritised Replay Buffer with N-step returns.
    """

    def __init__(
            self, 
            obs_shape: Tuple, 
            size: int, 
            batch_size: int = 32, 
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
        ):
        """
        Instantiates a PrioritisedReplayBuffer.
        """
        super().__init__(obs_shape, size, batch_size, n_step, gamma)
        self.alpha = alpha

        self.max_priority = 1
        self.tree_ptr = 0

        # The capacity must be positive and a power of 2.
        tree_capcity = 1
        while tree_capcity < self.max_size:
            tree_capcity *= 2

        self.sum_tree = SumSegmentTree(tree_capcity)
        self.min_tree = MinSegmentTree(tree_capcity)

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, next_obs: np.ndarray, done: bool):
        """
        Stores the experience and its associated priority.
        """
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """
        Samples a batch of experiences with prioritised replay.
        """
        indices = self._sample_proportional()
        weights = np.array([self._calculate_weight(idx, beta) for idx in indices])

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            weights=weights,
            indices=indices,
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Updates the priorities of sample transitions.
        """
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """
        Samples indices based on proportions.
        """
        indices = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices
    
    def _calculate_weight(self, idx: int, beta: float) -> float:
        """
        Calculates the weight of the experience at `idx`.
        """
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight
        
        return weight