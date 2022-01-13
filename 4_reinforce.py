import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import gym
import numpy as np

# make env, observation and action dimensions
env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# make a policy model
hidden_dim = 32
logits_net = nn.Sequential(
    nn.Linear(obs_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, act_dim))

def get_discounted_rewards(rews):
    GAMMA = 0.9 # discount factor
    discounted_rewards = []
    n = len(rews)
    for t in range(n):
        Gt = 0
        pw = 0
        for r in rews[t:]:
            Gt += GAMMA**pw * r
            pw += 1
        discounted_rewards.append(Gt)
    return (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)

# policy
def get_policy(obs):
    logits = logits_net(obs) # network outputs logits (-inf, +inf)
    return Categorical(logits=logits) # we turn them into probability p_i = [0, 1]

# action selection function
def get_action(obs):
    return get_policy(obs).sample().item()

# loss function = policy gradient (simplified expression, mean included)
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
    #return -(logp * weights).sum()

# setup hyperparameters
epochs = 50
batch_size = 5000
lr = 0.01

# setup optimizer
optimizer = torch.optim.Adam(logits_net.parameters(), lr=lr)

def train_one_epoch():
    # make some empty lists for logging
    batch_obs = []     # for observations
    batch_acts = []    # for actions
    batch_weights = [] # for R(tau) = weight in policy gradient
    batch_rets = []    # for measuring episode returns
    batch_lens = []    # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset() # first obs comes from starting distribution
    done = False      # signal from environment that episode is over
    ep_rews = []      # list of rewards acquired throughout episode

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:
        if not finished_rendering_this_epoch:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

        # save action and reward
        batch_acts.append(act)
        ep_rews.append(rew)

        # if episode is over
        if done:
            # record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a_t | s_t) is reward-to-go from t
            batch_weights += get_discounted_rewards(ep_rews).tolist()

            # reset episode specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of data
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.float32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                              )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens

# training loop
for i in range(epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
