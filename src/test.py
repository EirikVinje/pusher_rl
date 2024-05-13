import gymnasium as gym
import numpy as np




class Env:

    def __init__(self, 
                 n_states : int, 
                 env : str = "MountainCar-v0", 
                 epochs : int = 1):
        
        self.env = gym.make(env, render_mode="human")
        self.env.action_space.seed(42)

        self.n_states = n_states
        self.epochs = epochs
        self.n_actions = int(self.env.action_space.n)
        self.q_table = np.zeros((self.n_states, self.n_states, self.n_actions))


    def _obs_to_state(self, obs):
        
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_dx = (env_high - env_low) / n_states
        a = int((obs[0] - env_low[0])/env_dx[0])
        b = int((obs[1] - env_low[1])/env_dx[1])
        
        return a, b

    def _run_episode(self):

        action = self.env.action_space.sample()
        
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self.env.reset(seed=42)


    def run(self):

        for _ in range(self.epochs):
            
            obs, info = self.env.reset(seed=42)
            
            print(obs)
            
            for _ in range(500):
                self._run_episode()

            self.env.close()


if __name__ == "__main__":
    
    env = Env(10)

    env.run()