from network import Network
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from IPython.display import clear_output
import time

class PPO():
    def __init__(self, env, batch_size=32, gamma=0.95, verbose=False, plot_frequency=10, render_frequency=1):
        
        self._init_hyperparameters(batch_size=batch_size, gamma=gamma)
        
        self.verbose = verbose
        self.plot_frequency=plot_frequency
        self.render_frequency=render_frequency
        
      # Extract environment information
        self.env = env
        self.act_dims = env.action_space.n
        self.obs_dims = np.prod(env.observation_space.shape)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.actor = Network(self.obs_dims, self.act_dims).to(self.device)
        self.critic = Network(self.obs_dims, 1).to(self.device)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dims,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)
        
    def _init_hyperparameters(self, lr=0.0001, clip=0.3, gamma=0.95, updates_per_iteration=5, batch_size=32):
         # Default values for hyperparameters, will need to change later.
        self.batch_size = batch_size
        self.lr = lr
        self.clip = clip # As recommended by the paper
        self.gamma = gamma  # could reduce to 0.8
        self.updates_per_iteration = updates_per_iteration
        
        
        
    def get_action(self, obs):
                
        obs = torch.tensor(obs, dtype=torch.float)
        
        # Query the actor network for a mean action.
        mean = self.actor(obs)
        # Create a distribution based on your action space type
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            dist = Categorical(logits=mean)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # Assuming continuous action space, modify this based on your action space type
            dist = MultivariateNormal(mean, self.cov_mat)
        else:
            raise NotImplementedError("Handle other action space types if needed")

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if(self.verbose):
            print(int(action.detach().numpy()))
        return int(action.detach().numpy()), log_prob.detach()


    def train(self, timesteps_total):
        
        actor_losses = []
        critic_losses = []
        scores = []
        
        for frame_idx in range(1, timesteps_total + 1):        
            print("frame index: "+str(frame_idx))
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, score = self.rollout()
            
            #append the episode scores to scores
            scores+=score
            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            if len(batch_obs) > 0:
                # Calculate losses and update networks
                #what is this for loop for?
                for _ in range(self.updates_per_iteration):
                    # Calculate pi_theta(a_t | s_t)
                    V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                    # Calculate ratios
                    ratios = torch.exp(curr_log_probs - batch_log_probs)
                    # Calculate surrogate losses
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                    
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, batch_rtgs)
                    
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())

                    # Calculate gradients and perform backward propagation for actor 
                    # network
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    
                    # Calculate gradients and perform backward propagation for critic network    
                    self.critic_optim.zero_grad()    
                    critic_loss.backward()    
                    self.critic_optim.step()
                    
            self.clip*=0.997
            
            if frame_idx % self.render_frequency == 0:
                self.render_episode()
                
            if frame_idx % self.plot_frequency == 0:
                self._plot(frame_idx, scores, actor_losses, critic_losses)   
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            dist = Categorical(logits=mean)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # Assuming continuous action space, modify this based on your action space type
            dist = MultivariateNormal(mean, self.cov_mat)
        else:
            raise NotImplementedError("Handle other action space types if needed")
        
        
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and log probs log_probs
        return V, log_probs
    
    
    def _plot(self, frame_idx: int, scores: List[float], actor_losses: List[float], critic_losses: List[float]):
        """Plots the training progress."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        #rinstate below later
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('actor loss')
        plt.plot(actor_losses)
        plt.subplot(133)
        plt.title('critic loss')
        plt.plot(critic_losses)
        plt.show(block=False)
        # Automatically close the plot after 1 minute
        timeout = time.time() + 60  # 60 seconds = 1 minute
        while time.time() < timeout:
            plt.pause(1)

        plt.close()
        
    
    def rollout(self):
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        scores = []          
        
        # Number of timesteps run so far this batch
        runs = 0
        while runs < self.batch_size:
            runs+=1
            
            # Rewards this episode
            ep_rews = []
            state, _ = self.env.reset()
            
            state = state.flatten()
            done = False
            
            score = 0
            
            while not done:
                
                # Collect observation
                batch_obs.append(state)
                action, log_prob = self.get_action(state)
                _, reward, done = self.step(action)
            
                # Collect reward, action, and log prob
                score+=reward
                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
            # Collect episodic length and rewards
            batch_rews.append(ep_rews)
            scores.append(score)
        
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, scores

    def render_episode(self):
                
        state, _ = self.env.reset()
        state = state.flatten()
        done = False        
        self.verbose = False

        i = 0
        while not done:
            i+=1
            action, _ = self.get_action(state)
            print(action)
            state, _, done = self.step(action)
            clear_output(True)
            plt.imshow(self.env.render())
                    
        clear_output(True)
        plt.imshow(self.env.render())
        time.sleep(0.1)
        
        self.verbose = False
        
        plt.close()
        self.env.close()

    
    def step(self, action: np.ndarray):
        """
        Takes an action and returns the response of the env.
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = next_state.flatten()

        #if not self.is_test:

        return next_state, reward, done
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
import gymnasium as gym

print("running")
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.configure({
    "screen_width": 640,
    "screen_height": 480,
    "duration": 50,
    "vehicles_count": 50
})

seed = 777

def seed_torch(seed: int):
    torch.manual_seed(seed)

    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)

num_frames = 20_000
memory_size = 15_000
gamma = 0.95
batch_size = 10
target_update = 50

agent = PPO(env, batch_size=batch_size, gamma=gamma)
agent.train(num_frames)

