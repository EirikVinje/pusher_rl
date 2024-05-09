import gymnasium as gym

class DQLAgent:
   def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.shape
        self.EPISODES = 1000

   def build_model(self):
        pass

   def get_action(self, state):
        return env.action_space.sample()

   def replay(self, state, action, reward, next_state, done):
        pass

   def train(self):
        pass

   def run(self):
        pass

env = gym.make("Pusher-v4", render_mode="human")
observation, info = env.reset(seed=42)
agent = DQLAgent(env)

for _ in range(1000):
   action = agent.get_action(observation)
   observation, reward, terminated, truncated, info = env.step(action)
   print(observation)
   print(reward)

   if terminated or truncated:
      observation, info = env.reset()

env.close()