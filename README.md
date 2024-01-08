# Autonomous Control in a Highway Environment with DRL
You can find the highway environment [here](https://highway-env.farama.org/).

## This Branch
Contains implementations of the following algorithms based off of [Deep Q-Networks](https://arxiv.org/pdf/1312.5602.pdf):
- [DQN](base_dqn/dqn.ipynb)
- [Double DQN](double_dqn/double_dqn.ipynb)
- [DQN with Prioritised Experience Replay](per_dqn/prioritised_experience_replay.ipynb)
- [Dueling DQN](dueling_dqn/dueling_networks.ipynb)
- [DQN with Noisy Networks for Exploration](noisy_dqn/noisy_networks_for_exploration.ipynb)
- [Categorical DQN](categorical_dqn/categorical_dqn.ipynb)
- [DQN with N-step Learning](n_step_dqn/n_step_dqn.ipynb)
- [Rainbow DQN](rainbow_dqn/rainbow_dqn.ipynb)

We used [this](https://github.com/Curt-Park/rainbow-is-all-you-need) tutorial to help us implement the above algorithms.

## To run this code

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate your virtual environment (Unix):

```bash
source venv/bin/activate
```

Or, activate your virutal environment (Windows):

```bash
.\venv\bin\activate
```


Install the required packages:

```bash
pip install -r requirements.txt
```

And you should be set to run each of the notebooks! (as long as you've been blessed by the Cuda gods)