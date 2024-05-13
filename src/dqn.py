from collections import deque

import gymnasium as gym
import torch.nn as nn
import numpy as np
import torch


class ActorNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
            nn.Tanh() 
        )

    def forward(self, x):
        
        x = self.net(x)
        x = x * 2
        return x


class CriticNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs + n_outputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, u):

        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)


class Memory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, state_i1, action_i, state_i2, reward):
        self.memory.append([state_i1, action_i, state_i2, reward])
    
    def sample(self, batch_size):
        
        if batch_size > self.size():

            idx = np.random.choice(range(self.size()), self.size(), replace=False)
            return np.array(self.memory, dtype=object)[idx]
        
        idx = np.random.choice(range(self.size()), batch_size, replace=False)
        return np.array(self.memory, dtype=object)[idx]
    
    def size(self):
        return len(self.memory)
    

class Pusher:
    def __init__(self, seed, device : str = "cpu"):
        
        self.seed = seed
        self.r = 0.5
        self.device = device

        self.n_action = 7
        self.n_state = 23
        
        self.env = gym.make("Pusher-v4", render_mode="human")
        
        self.env.reset(seed=self.seed)

        self.actor_net = ActorNetwork(n_inputs=self.n_state, n_outputs=self.n_action)
        self.critic_net = CriticNetwork(n_inputs=(self.n_action + self.n_state), n_outputs=1)
        self.actor_target_net = ActorNetwork(n_inputs=self.n_state, n_outputs=self.n_action)
        self.critic_target_net = CriticNetwork(n_inputs=(self.n_action + self.n_state), n_outputs=1)

        self.memory = Memory(100)
    
    
    def action(self, state, steps):

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        action = self.actor_net(state).detach().numpy().reshape(7,)

        return action
        
    
    def step(self, batch_size):

        batch = self.memory.sample(batch_size)

        states = batch[:, 0]
        actions = batch[:, 1]
        next_ = batch[:, 2]
        rewards = batch[:, 3]


    
    def train(self, epochs : int = 10, steps : int = 100, lr : int = 0.001, batch_size : int = 4):

        actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(), lr=lr)
        critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=lr)
        critic_criterion = nn.MSELoss()

        for i in range(epochs):

            state, info = self.env.reset(seed=self.seed)
            
            for j in range(steps):
                
                action = self.action(state, steps)

                observation, reward, terminated, truncated, info = self.env.step(action)            

                if terminated:
                    next_state = None
                    self.env.reset(seed=self.seed)
                else:
                    next_state = observation

                self.memory.add(state, action, next_state, reward)

                state = next_state

                self.step(batch_size)

                # critic_loss = critic_criterion(predicted_q_values, target_q_values)
                # critic_optimizer.zero_grad()
                # critic_loss.backward()
                # critic_optimizer.step()
            
            print(batch[:, 3])





if __name__ == "__main__":
    
    pusher = Pusher(42)
    
    pusher.train(epochs=1, steps=5)

    