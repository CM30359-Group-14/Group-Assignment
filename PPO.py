from network import Network
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class PPO():
    def __init__(self, env):
        
        self._init_hyperparameters()
        
      # Extract environment information
        self.env = env
        self.act_dims = env.action_space.n
            
        self.obs_dims = env.observation_space.shape[0]
        self.actor = Network(self.obs_dims, self.act_dims)
        self.critic = Network(self.obs_dims, 1)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dims,), fill_value=0.5)
        
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)
        
    def _init_hyperparameters(self):
         # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600
        self.lr = 0.005
        self.clip = 0.2 # As recommended by the paper
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        
        
        
    def get_action(self, obs):
        
        obs_tensor = torch.tensor(obs[0], dtype=torch.float32)
        
        # Query the actor network for a mean action.
        mean = self.actor(obs_tensor)
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

        return action.detach().numpy(), log_prob.detach()


    def learn(self, timesteps_total):
        timesteps = 0
        
        while timesteps<timesteps_total:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            for _ in range(self.n_updates_per_iteration):
                    # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()
                
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            timesteps += np.sum(batch_lens)
    
    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs
    
    def rollout(self):
      # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []          
        batch_lens = []        
        
        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
            
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)     
            
        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
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


config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
}
env = gym.make('highway-v0', config={
    "action": {
        "type": "DiscreteMetaAction"
    }
})
env.configure(config)
model = PPO(env)
model.learn(10000)