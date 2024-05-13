from collections import deque
import argparse
import json
import os

import gymnasium as gym
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch


class OrnsteinUhlenbeckProcess:
    def __init__(self, size, theta=0.15, sigma=0.2):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.zeros(self.size)

    def sample(self):
        x = self.state
        dx = self.theta * (np.zeros(self.size) - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return dx


class ActorNetwork(nn.Module):
    def __init__(self, input_dim : int = 23, output_dim : int = 7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh() 
        )

    def forward(self, x):
        
        x = self.net(x)
        x = x * 2
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_dims : list[int, int] = [23, 7], output_dim : int = 7):
        super().__init__()

        self.state_input = nn.Linear(input_dims[0], 64)
        self.action_input = nn.Linear(input_dims[1], 64)

        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)

        self.output = nn.Linear(128, 1)

    def forward(self, state, action):
        
        state_features = nn.functional.relu(self.state_input(state))
        action_features = nn.functional.relu(self.action_input(action))

        x = torch.cat([state_features, action_features], dim=-1)
        
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))

        q_value = self.output(x)

        return q_value


class Memory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, state_i1, action_i, state_i2, reward):
        self.memory.append([state_i1, action_i, state_i2, reward])
    
    def sample(self, batch_size):
        
        if batch_size > self.size():
            return None
        
        idx = np.random.choice(range(self.size()), batch_size, replace=False)
        
        states = [self.memory[i][0] for i in idx]
        actions = [self.memory[i][1] for i in idx]
        next_states = [self.memory[i][2] for i in idx]
        rewards = [self.memory[i][3] for i in idx]
        
        not_terminated_idx = [i for i, x in enumerate(next_states) if x is not None]

        states = np.array(states, dtype=np.float32)[not_terminated_idx]
        actions = np.array(actions, dtype=np.float32)[not_terminated_idx]
        next_states = np.array(next_states, dtype=np.float32)[not_terminated_idx]
        rewards = np.array(rewards, dtype=np.float32)[not_terminated_idx]

        return [states, actions, next_states, rewards]
    
    def size(self):
        return len(self.memory)
    

class Pusher:
    def __init__(self, 
                 run_name : str,
                 save_n : int,
                 seed : int, 
                 device : str, 
                 epochs : int, 
                 batch_size : int,
                 lr : float = 0.001,
                 max_steps : int = 200, 
                 tau : float = 0.001, 
                 gamma : float = 0.90,
                 memory_size : int = 10_000,
                 checkpoint = None):
        
        self.save_n = save_n
        self.run_name = run_name
        self.seed = seed
        self.r = 0.5
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        self.n_action = 7
        self.n_state = 23
        
        self.env = gym.make("Pusher-v4", render_mode="rgb_array", max_episode_steps=max_steps)
        
        self.memory = Memory(memory_size)

        self.actor_net = ActorNetwork().to(self.device)
        self.critic_net = CriticNetwork().to(self.device)
        self.actor_target_net = ActorNetwork().to(self.device)
        self.critic_target_net = CriticNetwork().to(self.device)

        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=lr)
        self.critic_criterion = nn.MSELoss()

        self.noise_process = OrnsteinUhlenbeckProcess(size=self.n_action, theta=0.15, sigma=0.2)

        user = os.environ.get("USER")
        
        root = f"/home/{user}/data"
        if not os.path.exists(root):
            os.mkdir(root)
        
        pusherdir = os.path.join(root, "pusher_models")
        if not os.path.exists(pusherdir):
            os.mkdir(pusherdir)

        self.rundir = os.path.join(pusherdir, self.run_name)
        if not os.path.exists(self.rundir):
            os.mkdir(self.rundir)

        elif len(os.listdir(self.rundir)) != 0:
            assert False, "run dir already exists, create a new one or empty the old one"
        
        metafile = os.path.join(self.rundir, f"meta_{self.run_name}.json")
        if not os.path.exists(metafile):

            meta = {
                "seed" : self.seed,
                "n_action" : self.n_action,
                "n_state" : self.n_state,
                "device" : self.device,
                "save_n" : self.save_n,
                "epochs" : self.epochs,
                "batch_size" : self.batch_size             
            }

            with open(metafile, "w") as f:
                json.dump(meta, f)
    

    def _reset_env(self):

        state, _ = self.env.reset(seed=self.seed)
        return state


    def _action(self, state):

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        action = self.actor_net(state)
        
        if self.device == "cpu":
            action = action.detach().numpy().reshape(7,)
        elif self.device == "cuda":
            action = action.detach().cpu().numpy().reshape(7,)

        noise = self.noise_process.sample()
        action += noise
        action = np.clip(action, -2, 2)

        return action
        
    
    def _step(self):
        
        batch = self.memory.sample(self.batch_size)

        if batch is None:
            return 

        states, actions, next_states, rewards = batch

        # from memory
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        # pred Q
        pred_Q = self.critic_net(states, actions)
        
        # target Q
        with torch.no_grad():
            next_action = self.actor_target_net(next_states)
            target_Q = self.critic_target_net(next_states, next_action)

        target_Q = rewards + (self.gamma * target_Q)

        # critic loss 
        critic_loss = self.critic_criterion(pred_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # actor loss
        actor_loss = -self.critic_net(states, self.actor_net(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    

    def _save_model(self, epoch : int):
        
        model_path = os.path.join(self.rundir, f"{self.run_name}_{epoch}.pt")
        torch.save(self.actor_net.state_dict(), model_path)

    
    def train(self):

        for i in tqdm(range(self.epochs), desc="Episode"):

            state = self._reset_env()

            for j in range(10_000): # stops after set steps in declared env
                
                # get action
                action = self._action(state)

                # take action
                observation, reward, terminated, truncated, info = self.env.step(action)            

                # if terminated or truncated then reset
                if truncated or terminated:
                    state = self._reset_env()
                    next_state = None
                    break
                else:
                    next_state = observation

                # store in memory
                self.memory.add(state, action, next_state, reward)

                # move to next state
                state = next_state

                # do a step, if size of memory >= batch size
                self._step()

                # update critic target networks
                target_params = self.critic_target_net.state_dict()
                current_params = self.critic_net.state_dict()
                for name, param in target_params.items():
                    param.data.copy_(self.tau * current_params[name].data + (1 - self.tau) * param.data)

                # update actor target networks
                target_params = self.actor_target_net.state_dict()
                current_params = self.actor_net.state_dict()
                for name, param in target_params.items():
                    param.data.copy_(self.tau * current_params[name].data + (1 - self.tau) * param.data)
            
            self.noise_process.reset()

            if i % self.save_n == 0 and i != 0:
                self._save_model(epoch=i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save_n", type=int)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()

    seed = args.seed # which seed (track)
    device = args.device # cpu or cuda
    epochs = args.epochs # episodes
    save_n = args.save_n # how often to save
    run_name = args.run_name # name of run directory to save
    batch_size = args.batch_size # batch size
    
    pusher = Pusher(seed=seed, 
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    run_name=run_name, 
                    save_n=save_n)
    
    pusher.train()

    