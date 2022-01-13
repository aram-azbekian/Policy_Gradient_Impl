import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import numpy as np

# make env, observation and action dimensions
env = gym.make('CartPole-v0')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# hyperparameters
hidden_dim = 256
lr = 0.01

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        #state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

# initialize model and setup optimizer
actor_critic = ActorCritic(obs_dim, act_dim, hidden_dim)
ac_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

# setup hyperparameters
epochs = 50
num_steps = 5000
GAMMA = 0.99

for epoch in range(epochs):
    # batch variables for loss computation and logging
    batch_rets     = [] # episode returns over 1 batch
    batch_lens     = [] # episode lengths
    batch_Qvals    = [] # Q-values calculated over each step of each episode
    batch_values   = [] # outputs of V(s_t) (we need them to calculate adv. function)
    batch_acts     = [] # actions taken from the same policy
    batch_obs      = [] # observations

    # episode-specific variables
    rewards = [] # list of rewards acquired throughout episode
    values  = [] # list of Value-function outputs throughout episode

    # first observation comes from starting distribution
    obs = env.reset()

    # render first episode of each epoch
    finished_rendering_this_epoch = False if (epoch+1) % 50 == 0 else True

    # collect experience by acting in the environment with current policy
    while True:
        if not finished_rendering_this_epoch:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        value, policy_dist = actor_critic(obs)
        #value = value.detach().numpy()[0, 0]
        value = value[0, 0]
        #probs = policy_dist.detach()
        probs = policy_dist
        sampler = Categorical(probs)

        act = sampler.sample()
        log_prob = sampler.log_prob(act)
        obs, rew, done, _ = env.step(act.item())

        # save action, reward, value and action
        rewards.append(rew)
        values.append(value)
        batch_acts.append(act)

        # if episode is over
        if done:
            # we need to sample Qval here to calculate:
            # V^*(s_t) = [r_t + GAMMA * V^*(s_t+1)]
            Qval, _ = actor_critic(obs)
            Qval = Qval[0, 0]

            # compute Q-values
            Qvals = np.zeros(len(values))
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval
            batch_Qvals += Qvals.tolist()

            # append current values in the batch
            batch_values += values
            batch_lens.append(len(rewards))
            batch_rets.append(sum(rewards))

            # reset episode specific variables
            obs = env.reset()
            rewards = []
            values = []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough data
            if len(batch_obs) > num_steps:
                break

    # update actor-critic
    batch_values = torch.FloatTensor(batch_values)
    batch_Qvals = torch.FloatTensor(batch_Qvals)
    batch_acts = torch.FloatTensor(batch_acts)
    _, probs = actor_critic(np.array(batch_obs))
    batch_logprobs = Categorical(probs).log_prob(batch_acts)

    advantage = batch_Qvals - batch_values
    actor_loss = (-batch_logprobs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    ac_loss = actor_loss + critic_loss

    # take a single policy gradient update step
    ac_optimizer.zero_grad()
    ac_loss.backward()
    ac_optimizer.step()

    if epoch % 1 == 0:
        print(f'Epoch {epoch}: \t avg. return: {np.mean(batch_rets):.3f} \t avg. ep. length: {np.mean(batch_lens):.3f}')



