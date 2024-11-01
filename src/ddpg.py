from collections import deque
import argparse
import time
import json
import csv
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
    def __init__(self, input_dims : list[int, int] = [23, 7]):
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

    def add(self, state, action, next_state, reward):
        self.memory.append([state, action, next_state, reward])
    
    def sample(self, batch_size):
        
        if batch_size > self.size():
            return None

        idx = np.random.choice(range(self.size()), batch_size, replace=False)

        batch = [self.memory[i] for i in idx]

        not_terminated_idx = [i for i, x in enumerate(batch) if x[2] is not None]
        
        batch = [[batch[i][j] for i in not_terminated_idx] for j in range(4)]

        states, actions, next_states, rewards = batch

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        return [states, actions, next_states, rewards]
        

    def size(self):
        return len(self.memory)
    

class Pusher:
    def __init__(self, 
                 run_name : str,
                 epochs : int, 
                 batch_size : int,
                 memory_size : int,
                 max_episode_steps : int,
                 lr : float = 0.0001,
                 tau : float = 0.001, 
                 gamma : float = 0.90,
                 device : str = "cuda",
                 record : bool = True
                ):
        
        self.run_name = run_name
        self.device = device
        self.record = record
        
        self.batch_size = batch_size
        self.epochs = epochs
    
        self.n_action = 7
        self.n_state = 23
        
        self.max_episode_steps = max_episode_steps

        self.env = gym.make("Pusher-v4", render_mode="rgb_array", max_episode_steps=self.max_episode_steps)
        
        self.memory = Memory(memory_size)

        self.actor_net = ActorNetwork(input_dim=self.n_state, output_dim=self.n_action).to(self.device)
        self.critic_net = CriticNetwork(input_dims=[self.n_state, self.n_action]).to(self.device)
        self.actor_target_net = ActorNetwork(input_dim=self.n_state, output_dim=self.n_action).to(self.device)
        self.critic_target_net = CriticNetwork(input_dims=[self.n_state, self.n_action]).to(self.device)

        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=lr)
        self.mseloss = nn.MSELoss()

        self.ornstein = OrnsteinUhlenbeckProcess(size=self.n_action)

        if self.record:
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
                    "device" : self.device,
                    "epochs" : self.epochs,
                    "batch_size" : self.batch_size,
                    "max_episode_steps" : max_episode_steps,
                    "lr" : lr,
                    "tau" : tau,
                    "gamma" : gamma,
                    "memory_size" : memory_size
                }

                with open(metafile, "w") as f:
                    json.dump(meta, f)
    

    def _reset_env(self, seed:int =-1):
        
        if seed == -1:
            state, _ = self.env.reset()
            return state
        else:
            state, _ = self.env.reset(seed=seed)
            return state


    def action(self, state, add_noise : bool):

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor_net.eval()
        action = self.actor_net(state)
        self.actor_net.train()
        
        if self.device == "cpu":
            action = action.detach().numpy().reshape(7,)
        elif self.device == "cuda":
            action = action.detach().cpu().numpy().reshape(7,)

        if add_noise:
            noise = self.ornstein.sample()
            action += noise
            action = np.clip(action, -2, 2)
            
        return action

        
    def do_step(self, batch):

        states, actions, next_states, rewards = batch

        # from memory
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        # target Q
        with torch.no_grad():
            next_action = self.actor_target_net(next_states)
            target_Q = self.critic_target_net(next_states, next_action)

        target_Q = rewards + (self.gamma * target_Q)

        # critic loss 
        self.critic_optimizer.zero_grad()
        pred_Q = self.critic_net(states, actions)
        critic_loss = self.mseloss(pred_Q, target_Q)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # actor loss
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic_net(states, self.actor_net(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()
        

    def _save_model(self, epoch : int):
        
        model_path = os.path.join(self.rundir, f"{self.run_name}_{epoch}.pt")
        torch.save(self.actor_net.state_dict(), model_path)

    
    def _init_csv(self):

        self.csv_file_per_ep = os.path.join(self.rundir, f"{self.run_name}_metrics_per_ep.csv")

        with open(self.csv_file_per_ep, 'w', newline='') as file:  
            writer = csv.writer(file)
            headers = ['actor_loss', 'critic_loss', 'rewards'] 
            writer.writerow(headers)
        
        self.csv_file_rewards = os.path.join(self.rundir, f"{self.run_name}_metrics_rewards.csv")

        with open(self.csv_file_rewards, 'w', newline='') as file:  
            writer = csv.writer(file)
            headers = ['reward', 'episode'] 
            writer.writerow(headers)
        
        self.csv_file_dt = os.path.join(self.rundir, f"{self.run_name}_metrics_time.csv")

        with open(self.csv_file_dt, 'w', newline='') as file:  
            writer = csv.writer(file)
            headers = ['time', 'episode'] 
            writer.writerow(headers)

    
    def _write_csv_per_ep(self, actor_loss, critic_loss, reward):

        with open(self.csv_file_per_ep, 'a', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow([actor_loss, critic_loss, reward])


    def _write_csv_reward(self, reward, episode):

        with open(self.csv_file_rewards, 'a', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow([reward, episode])
    

    def _write_csv_time(self, delta_time, episode):

        with open(self.csv_file_dt, 'a', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow([delta_time, episode])


    def evaluate(self):
        
        state = self._reset_env()
        while True:

            action = self.action(state, add_noise=False)
            state, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                return reward


    def train(self):

        if self.record:
            self._init_csv()

        best_reward = -100000
        best_step = -1
        actor_loss = 0
        critic_loss = 0
        
        state = self._reset_env()
        self.reward_que = deque([], maxlen=10)

        with tqdm(total=self.epochs) as bar:
            
            bar.set_description("best : (0, {}), p.e : (0.0)".format(best_reward))
            
            for i in range(self.epochs):
                
                start_t = time.time()
                
                action = self.action(state, add_noise=True)

                observation, reward, terminated, truncated, info = self.env.step(action)            

                if truncated or terminated:
                    next_state = None
                else:
                    next_state = observation

                self.memory.add(state, action, next_state, reward)

                state = next_state
                
                batch = self.memory.sample(self.batch_size)
                
                if batch is not None:
                    
                    al, cl = self.do_step(batch)
                    actor_loss = al
                    critic_loss = cl

                    target_params = self.critic_target_net.state_dict()
                    current_params = self.critic_net.state_dict()
                    for name, param in target_params.items():
                        param.data.copy_(self.tau * current_params[name].data + (1 - self.tau) * param.data)

                    target_params = self.actor_target_net.state_dict()
                    current_params = self.actor_net.state_dict()
                    for name, param in target_params.items():
                        param.data.copy_(self.tau * current_params[name].data + (1 - self.tau) * param.data)

                if i % 10 == 0:
                    self.reward_que.append(self.evaluate())

                if i % 100 == 0:
                    reward = float(np.mean(self.reward_que))

                    if reward > best_reward:
                        best_reward = reward
                        best_step = i  
                    
                    if self.record:
                        self._write_csv_reward(best_reward, i)
                        self._write_csv_per_ep(actor_loss, critic_loss, reward)
                        
                end_t = time.time()

                bar.set_description("best : ({}, {}) p.e : ({}s)".format(best_step, round(best_reward,2), round(end_t-start_t,3)))
                bar.update(1)

                if state is None:   
                    state = self._reset_env()
                    
        return best_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--memory", type=int)
    parser.add_argument("--max_episode_steps", type=int)
    parser.add_argument("--record", type=int)

    args = parser.parse_args()

    device = args.device # cpu or cuda
    epochs = args.epochs # episodes
    run_name = args.run_name # name of run directory to save
    batch_size = args.batch_size # batch size
    memory = args.memory # memory size
    max_episode_steps = args.max_episode_steps # max steps per episode
    record = args.record
    
    pusher = Pusher(device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    run_name=run_name, 
                    memory_size=memory,
                    max_episode_steps=max_episode_steps,
                    record=record)
    
    pusher.train()
