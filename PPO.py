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
import csv

"""
based partially on Eric Yang Yu's Medium guide to implementing PPO found here https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""


class PPO():
    
    def __init__(self, env, lr=0.2, clip=0.3,batch_size=32, gamma=0.95, verbose=False, plot_frequency=10, render_frequency=1):
        
        self._init_hyperparameters(lr=lr, batch_size=batch_size, gamma=gamma, clip=clip)
        
        self.verbose = verbose
        self.plot_frequency=plot_frequency
        self.render_frequency=render_frequency
        self.save_file = "scores2.csv"
        
        self.steps = 0
        
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
        
    def _init_hyperparameters(self, lr=0.0001, clip=0.35, gamma=0.95, updates_per_iteration=10, batch_size=32):
         # Default values for hyperparameters, will need to change later.
        self.batch_size = batch_size
        self.lr = lr
        self.clip = clip # As recommended by the paper
        self.gamma = gamma  # could reduce to 0.8
        self.updates_per_iteration = updates_per_iteration      
        
    
    def get_action(self, obs):
        """
        generates the next action based on current observation

        Args:
            obs (tensor): current observation

        Raises:
            NotImplementedError: if distribution isn't of correct type (not needed but no need to remove it either)

        Returns:
            action, log_probs (tuple): the chosen action, log probabilities of each action
        """
                
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
            if (int(action.detach().numpy())==3):
                print("speedup")
                
        return int(action.detach().numpy()), log_prob.detach()


    def train(self, timesteps_total):
        """
        training loop for agent. 

        Args:
            timesteps_total (int): the total number of timesteps to train the agent for
        """
        
        #actor and critic losses, alongside an array of scores for plotting
        actor_losses = []
        critic_losses = []
        scores = []
        
        #used to count number of batches performed
        loop_num = 0
        
        while(self.steps<timesteps_total):
            loop_num+=1
            #really just to check it hasn't crashed
            print("loop "+str(loop_num))
            
            
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, score = self.collect_batch()
            
            #append the batch scores to scores
            scores+=score
            # Calculate V.
            V, _ = self.evaluate(batch_obs, batch_acts)
            
            # Calculate advantage and normalise advantages
            advantages = batch_rtgs - V.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            
            
            # Calculate losses and update networks
            for _ in range(self.updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # Calculate surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
                    
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
            
            #saves scores to csv         
            with open(self.save_file, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for item in score:
                    csv_writer.writerow([item])
            
            #renders an episode whenever the number of batches is divisible by render frequency
            if loop_num % self.render_frequency == 0:
                self.render_episode()
                
            #renders a plot whenever the number of batches is divisible by plot frequency 
            if loop_num % self.plot_frequency == 0:
                self._plot(loop_num, scores, actor_losses, critic_losses)   
    
    
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
        
    
    def collect_batch(self):
        """
        collects a batch of episodes

        Returns:
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, scores: tuple of batch data arrays
        """
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
        #compute rewards to go of batch
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, scores

    def render_episode(self):
        """
        renders a single episode of the agent at time of calling

        Returns:
            total_reward (double): total reward for that episode
        """
        
        state, _ = self.env.reset()
        state = state.flatten()
        done = False        
        self.verbose = False
        total_reward = 0

        i = 0
        while not done:
            i+=1
            #gets the action
            action, _ = self.get_action(state)
            print(action)
            #steps environment with action
            state, reward, done = self.step(action)
            clear_output(True)
            #shows the environment
            plt.imshow(self.env.render())
            total_reward+=reward
                    
        clear_output(True)
        plt.imshow(self.env.render())
        time.sleep(0.1)
        
        self.verbose = False
        
        #closes the plot and env
        plt.close()
        self.env.close()
        
        #returns the total reward for the episode
        return total_reward

    
    def step(self, action: np.ndarray):
        """
        Takes an action and returns the response of the env.
        """
        self.steps+=1
        #prints the number of timesteps trained for every 1000 timesteps (mostly so I know it's still running)
        if(self.steps%1000==0):
            print("current frame/timestep count:"+str(self.steps))
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = next_state.flatten()

        #if not self.is_test:

        return next_state, reward, done
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain the same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.append(discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def test(self, test_num):
        """
        test agent on test_num episodes and collect the average reward

        Args:
            test_num (int): number of episodes to test on

        Returns:
            average_reward (double): the average reward of the test
        """
        average = 0
        for i in range(test_num):
            average+=self.render_episode()
        
        return average/test_num


#imports the environment
import gymnasium as gym

#build env with appropriate config
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.configure({
    'duration': 50,
    'lanes_count': 4
    
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

num_frames = 250_000
gamma = 0.85
batch_size = 64

#initialise, train and test agent.
agent = PPO(env, lr=0.005, batch_size=batch_size, gamma=gamma, clip=0.14)
agent.train(num_frames)
print("reward = "+str(agent.test(1000)))

