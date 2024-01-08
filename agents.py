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
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output

from abc import abstractmethod, ABC

from buffers import ReplayBuffer, NStepReplayBuffer, PrioritisedReplayBuffer
from networks import Network, DuelingNetwork, CategoricalNetwork, NoisyNetwork, RainbowNetwork


class BaseAgent(ABC):
    """
    Abstract Base Class for RL agents.
    """

    def __init__(self, env: gym.Env):
        self.env = env

        # Mode: train/test.
        self.is_test = False

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
    def train(self, num_frames: int, plotting_interval: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trains the agent for a specified number of frames.

        :return: a tuple of training metrics (scores, losses, epsilons)
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, model_path: str):
        """
        Saves the model's parameters to disk at `model_path`.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls: Self, model_path: str, **kwargs) -> Self:
        """
        Loads a saved Agent from disk.
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
        super().__init__(env)

        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.memory_size = memory_size

        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.n

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

        return (scores, losses, epsilons)

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
    

class MlpDQNWithPERAgent(MlpDQNAgent):
    """
    Class representing a DQN with Prioritised Experience Replay Agent.
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        # PER Parameters.
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
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

        # Prioritised Experience Replay settings.
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritisedReplayBuffer(self.obs_shape, memory_size, batch_size, alpha)
    
    def update_model(self) -> torch.Tensor:
        """
        Updates the model by gradient descent.
        """
        # PER uses beta to calculate weights.
        samples = self.memory.sample_batch(self.beta)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # PER: Calculate importance sampling before the average.
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # PER: Update experience priorities.
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """
        Trains the agent.
        """
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)

        update_count = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if done:
                # The episode has ended.
                state, _ = self.env.reset(seed=self.seed)
                
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

            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

        return scores, losses, epsilons

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

        # Calculate element-wise DQN loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss
    

class MlpDuelingDQNAgent(MlpDQNAgent):
    """
    Class representing a DQN agent with Dueling Networks.
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        seed: int,
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
        self.dqn = DuelingNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target = DuelingNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.dqn.parameters())
    
    def update_model(self) -> torch.Tensor:
        """
        Updates the model by gradient descent.
        """
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)
        
        self.optimiser.zero_grad()
        loss.backward()

        # Dueling DQN: We clip the gradients to have their norm less than or equal to 10.
        clip_grad_norm_(self.dqn.parameters(), 10.0)

        self.optimiser.step()

        return loss.item()
    

class MlpCategoricalDQNAgent(MlpDQNAgent):

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
        # Categorical DQN Parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
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
            gamma,
        )

        # Categorical DQN Parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            v_min, v_max, atom_size
        ).to(self.device)

        # Categorical Networks
        self.dqn = CategoricalNetwork(
            self.obs_dim, self.action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target = CategoricalNetwork(
            self.obs_dim, self.action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.dqn.parameters())

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        device = self.device

        # Shape = (batch_size, obs dim 1, obs dim 2, ...)
        # This flattens the observation dimensions of `state` and `next_state`.
        state = torch.FloatTensor(samples["obs"].reshape(self.batch_size, -1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"].reshape(self.batch_size, -1)).to(device)

        # Reshapes each 1-dimesional array into a 2-dimensional array with one column.
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm.
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.dqn_target(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])

        loss = -(proj_dist * log_p).sum(1).mean()

        return loss
    

class MlpNoisyDQNAgent(MlpDQNAgent):

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
    ):
        super().__init__(
            env, 
            memory_size, 
            batch_size, 
            target_update, 
            seed=seed, 
            gamma=gamma
        )
        # Noisy DQN Networks.
        obs_dim = np.prod(self.obs_shape)
        self.dqn = NoisyNetwork(obs_dim, self.action_dim).to(self.device)
        self.dqn_target = NoisyNetwork(obs_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.dqn.parameters())

    def select_action(self, state: np.ndarray, determinstic: bool = False) -> np.ndarray:
        # As we are using noisy networks, we do not use an epsilon-greedy policy for
        # action selection.
        flattened_state = state.flatten()

        selected_action = self.dqn(
            torch.Tensor(flattened_state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action
    
    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Reset the noise in the noisy network layers.
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
    

class NStepDQNAgent(MlpDQNAgent):

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
        # N-step Learning
        n_step: int = 3,
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

        # Memory for 1-step learning.
        self.memory = NStepReplayBuffer(
            self.obs_shape, memory_size, batch_size, n_step=1, gamma=gamma
        )

        # Memory for N-step learning.
        self.use_n_step = n_step > 1
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = NStepReplayBuffer(
                self.obs_shape, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # Add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        loss = self._compute_dqn_loss(samples, self.gamma)

        # N-step Learning loss. We combine 1-step and n-step loss so as to
        # prevent high variance.
        if self.use_n_step:
            samples = self.memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma ** self.n_step
            n_loss = self._compute_dqn_loss(samples, gamma)
            loss += n_loss

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()
    

class RainbowDQNAgent(MlpDQNAgent):

    def __init__(
        self,        
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int = 42,
        gamma = 0.99,
        # Priortised Experience Replay Params
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN Params
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        super().__init__(
            env,
            memory_size,
            batch_size,
            target_update,
            seed=seed,
            gamma=gamma,
        )

        # Prioritised Replay Buffer. This is the memory for 1-step learning.
        self.beta = beta
        self.alpha = alpha
        self.prior_eps = prior_eps
        self.memory = PrioritisedReplayBuffer(
            self.obs_shape, memory_size, batch_size, alpha=alpha, gamma=gamma
        )

        # Memory for N-step learning.
        self.use_n_step = n_step > 1
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = NStepReplayBuffer(
                self.obs_shape, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN Parameters.
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            v_min, v_max, atom_size
        ).to(self.device)

        # Value networks
        self.dqn = RainbowNetwork(
            self.obs_dim, self.action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target = RainbowNetwork(
            self.obs_dim, self.action_dim, atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # Optimiser
        self.optimiser = optim.Adam(self.dqn.parameters())

    def select_action(self, state: np.ndarray, determinstic: bool = False) -> np.ndarray:
        # As we are using noisy networks, we do not use an epsilon-greedy policy for
        # action selection.
        flattened_state = state.flatten()

        selected_action = self.dqn(
            torch.Tensor(flattened_state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # Add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done
    
    def update_model(self) -> Tensor:
        # PER uses beta to calculate the weights.
        samples = self.memory.sample_batch(self.beta)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step learning loss.
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # N-step learning loss. We combine 1-step and n-step loss so as to
        # prevent high variance. The original RainbowDQN employs only n-step.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)

            elementwise_loss_n = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n

        # PER: Importance sampling before averaging.
        loss = torch.mean(elementwise_loss * weights)

        self.optimiser.zero_grad()
        loss.backward()

        # Dueling Networks: We clip the gradients to prevent exploding gradients.
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimiser.step()

        # PER: Update experience priorities.
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # Noisy Network: Reset the noise in the noisy network layers.
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
    
    def train(self, num_frames: int, plotting_interval: int = 200):
        """
        Trains the agent.
        """
        self.is_test = False
        state, _ = self.env.reset()

        update_count = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(num_frames):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: Increase param beta.
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

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

                # Noisy Networks: Removed decrease of epsilon.

                # If a hard update of the target network is needed.
                if update_count % self.target_update == 0:
                    self._target_hard_update()

            if (frame_idx + 1) % plotting_interval == 0:
                self._plot(frame_idx + 1, scores, losses, [])

        self.env.close()

        return scores, losses, []

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float = None) -> torch.Tensor:
        """
        Computes and returns the Categorical DQN loss.
        """
        if gamma is None:
            gamma = self.gamma

        device = self.device
        # Shape = (batch_size, obs dim 1, obs dim 2, ...)
        # This flattens the observation dimensions of `state` and `next_state`.
        state = torch.FloatTensor(samples["obs"].reshape(self.batch_size, -1)).to(device)
        next_state = torch.FloatTensor(samples["next_obs"].reshape(self.batch_size, -1)).to(device)

        # Reshapes each 1-dimesional array into a 2-dimensional array with one column.
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )
        
        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss