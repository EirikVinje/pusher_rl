import os
import time
import argparse

import gymnasium as gym
import numpy as np
import torch

from ddpg import ActorNetwork

class Model:
    def __init__(self, path_model : str, render : int, max_steps : int, device="cpu"):
        
        if render:
            render = "human"
        else:
            render = "rgb_array"

        self.env = gym.make("Pusher-v4", render_mode=render, max_episode_steps=max_steps)

        self.device = device

        self.net = ActorNetwork().to(self.device)
        self.net.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
        self.net.eval()

    def test(self, seeds : list[int], show_progress : bool = False):

        rewards = []
        for seed in seeds:
            
            state, info = self.env.reset(seed=int(seed))
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            i = 0
            rewards_i = []
            while True:

                action = self.net(state)

                if self.device == "cpu":
                    action = action.detach().numpy().reshape(7,)
                elif self.device == "cuda":
                    action = action.detach().cpu().numpy().reshape(7,)

                state, reward, terminated, truncated, info = self.env.step(action)
                rewards_i.append(reward)

                if show_progress:
                    print(f"seed: {seed} :: Reward at step {i}: {reward}", end="\r")
                
                if terminated or truncated:
                    break

                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                i += 1
            
            if show_progress:
                print()

            rewards.append(rewards_i)

        self.env.close()

        return rewards


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    print(args)

    model = Model(args.model, args.render)
    model.test()
