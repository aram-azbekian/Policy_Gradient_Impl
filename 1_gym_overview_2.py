import gym

env = gym.make('CartPole-v0') # creating gym environment

for episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print(observation)
        if done:
            print(f'Episode {episode+1} finished after {t+1} steps.')
            break
env.close()
