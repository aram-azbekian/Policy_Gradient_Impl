import gym

env = gym.make('CartPole-v0') # creating gym environment
env.reset() # setting it to initial state

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # always taking random action from env
env.close()
