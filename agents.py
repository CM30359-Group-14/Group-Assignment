import os
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch._tensor import Tensor

from typing import Dict, List, Tuple, Self
from IPython.display import clear_output

from abc import abstractmethod, ABC

from buffers import ReplayBuffer
from networks import Network


class BaseAgent(ABC):
    """
    Abstract Base Class for RL agents.
    """

    def set_mode(self, is_test: bool):
        """
        Sets the inference mode for the agent.
        """
        self.is_test = is_test

    def predict(self, state: np.ndarray, determinstic: bool = True) -> np.ndarray:
        """
        Selects an action from the input state using a (potentially) epsilon-greedy policy.
        """
        return self.select_action(state, determinstic)
    
    @abstractmethod
    def select_action(self, state: np.ndarray, determinstic: bool = False) -> np.ndarray:
        """
        Selects an action from the input state using an epsilon-greedy policy.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def train(self, num_frames: int, plotting_interval: int = 200):
        """
        Trains the agent for a specified number of frames.
        """
        raise NotImplementedError()
    
    @staticmethod
    def _init_seed(seed: int):
        """
        Initialises the seed used by random number generators.
        """
        torch.manual_seed(seed)

        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(seed)


class BaseDQNAgent(BaseAgent):
    """
    Abstract Base class for DQN agents.
    """

    def __init__(
        self,        
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 1e-4,
        seed: int = 42,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.memory_size = memory_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # DQN Networks (to be instantiated)
        self.dqn = None
        self.dqn_target = None
        self.optimiser = None

        # Experience Replay Buffer.
        self.memory = ReplayBuffer(self.obs_shape, memory_size, batch_size)
        # The next transition to store in memory.
        self.transition = list()

        # Mode: train/test.
        self.is_test = False

        self._init_seed(seed)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """
        Takes an action and returns the response of the env.
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        """
        Updates the model parameters by gradient descent.
        """
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state, _ = self.env.reset()

        update_count = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(num_frames):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            if done:
                # The episode has ended.
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                # Training is ready once the replay buffer contains enough transition samples.
                loss = self.update_model()
                losses.append(loss)
                update_count += 1

                # Linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon,
                    self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                # If a hard update of the target network is needed.
                if update_count % self.target_update == 0:
                    self._target_hard_update()

            if (frame_idx + 1) % plotting_interval == 0:
                self._plot(frame_idx + 1, scores, losses, epsilons)

        self.env.close()

    def test(self, num_episodes: int, render: bool = True, time_interval: float = 0.2) -> Tuple[List, List]:
        self.is_test = True

        episode_lengths = []
        undiscounted_rewards = []
        for _ in range(num_episodes):
            done = truncated = False

            episode_reward = 0
            episode_length = 0

            obs, _ = self.env.reset()
            while not (done or truncated):
                action = self.predict(obs, True)
                obs, reward, done, truncated, _ = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if not render:
                    continue

                clear_output(True)
                plt.imshow(self.env.render())
                plt.show()
                time.sleep(time_interval)

            undiscounted_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return episode_lengths, undiscounted_rewards
    
    def save(self, model_path: str):
        """
        Saves the Deep Q-Network model parameters to disk at `model_path`.
        """
        if not os.path.exists("./model/"):
            os.mkdir("model")

        torch.save(self.dqn.state_dict(), model_path)

    @classmethod
    def load(
        cls, 
        model_path: str,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        **kwargs
    ) -> Self:
        agent = cls(env, memory_size, batch_size, target_update, **kwargs)

        agent.dqn.load_state_dict(torch.load(model_path))
        agent.dqn_target.load_state_dict(torch.load(model_path))
        
        # agent.dqn.eval()
        agent.dqn_target.eval()

        return agent
    
    @abstractmethod
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float = None) -> torch.Tensor:
        """
        Computes and returns the DQN loss.
        """
        raise NotImplementedError()

    def _target_hard_update(self):
        """
        Performs a hard update of the target network: target <- behavioural.
        """
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plots the training progress."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-25:])))
        # Plot the rolling mean score of the last 25 episodes.
        plt.plot(self._calculate_rolling_mean(scores, 25))
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses[20:])
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()

    @staticmethod
    def _calculate_rolling_mean(data: List, window_size: int) -> np.ndarray:
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='valid')


class MlpDQNAgent(BaseDQNAgent):
    """
    Class representing a DQN agent utilising MLP Feed-Forward Neural Networks.
    """

    def __init__(
        self,        
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 1e-4,
        seed: int = 42,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__(
            env, 
            memory_size, 
            batch_size, 
            target_update, 
            epsilon_decay, 
            seed, 
            max_epsilon, 
            min_epsilon, 
            gamma
        )

        # Networks: DQN behaviour network, DQN target network
        self.obs_dim = np.prod(self.obs_shape)
        self.dqn = Network(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target = Network(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.dqn.parameters())

    def select_action(self, state: np.ndarray, determinstic: bool = False) -> np.ndarray:
        """
        Selects an action from the input state using an epsilon-greedy policy.
        """
        if not determinstic and np.random.random() < self.epsilon:
            selected_action = self.env.action_space.sample()
        else:
            flattened_state = state.flatten()
            selected_action = self.dqn(
                torch.FloatTensor(flattened_state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float = None) -> torch.Tensor:
        """
        Computes and returns the DQN loss.
        """
        if gamma is None:
            gamma = self.gamma

        device = self.device
        # Shape = (batch_size, obs dim 1, obs dim 2, ...)
        # This flattens the observation dimensions of `state` and `next_state`.
        state = torch.FloatTensor(samples["obs"].reshape(self.batch_size, -1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"].reshape(self.batch_size, -1)).to(device)

        # Reshapes each 1-dimesional array into a 2-dimensional array with one column.
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t = r + gamma * v(s_{t+1}) if state != terminal
        #     = r                      otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(device)

        # Calculate DQN loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss


class MlpDDQNAgent(MlpDQNAgent):
    """
    Class representing a Double DQN agent utilising MLP Feed-Forward Neural Networks.
    """

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> Tensor:
        """
        Computes and returns the Double DQN loss.
        """
        device = self.device

        # Shape = (batch_size, obs dim 1, obs dim 2, ...)
        # This flattens the observation dimensions of `state` and `next_state`.
        state = torch.FloatTensor(samples["obs"].reshape(self.batch_size, -1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"].reshape(self.batch_size, -1)).to(device)

        # Reshapes each 1-dimesional array into a 2-dimensional array with one column.
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t = r + gamma * v(s_{t+1}) if state != terminal
        #     = r                      otherwise
        curr_q_value = self.dqn(state).gather(1, action)

        # This line is what makes the agent a Double DQN agent.
        next_q_value = self.dqn_target(next_state).gather( 
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        
        mask = 1 - done
        # Calculate the TD target
        target = (reward + self.gamma * next_q_value * mask).to(device)

        # Calculate DQN loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss