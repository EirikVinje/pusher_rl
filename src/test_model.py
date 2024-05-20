import os
import argparse

import gymnasium as gym
import numpy as np
import torch

from ddpg import ActorNetwork

class Model:
    def __init__(self, path_model, device="cpu"):
        
        user = os.environ.get("USER")
        root = f"/home/{user}/data/pusher_models/"
        path_model = os.path.join(root, path_model)

        self.max_steps = 200

        self.env = gym.make("Pusher-v4", render_mode="human", max_episode_steps=self.max_steps)

        self.device = device

        self.net = ActorNetwork().to(self.device)
        self.net.load_state_dict(torch.load(path_model,map_location=torch.device('cpu')))
        self.net.eval()

    def test(self):

        stest = [710, 0, 5, 11, 14, 23]
        stest = [3559, 3216, 7890, 5242, 4924, 3588, 722, 8119]

        seeds = np.random.choice(10000, 8, replace=False)
        # seeds = range(100)

        seeds = stest
        
        for seed in seeds:
            
            print(seed)

            state, info = self.env.reset(seed=int(seed))
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
 
            while True:

                action = self.net(state)

                if self.device == "cpu":
                    action = action.detach().numpy().reshape(7,)
                elif self.device == "cuda":
                    action = action.detach().cpu().numpy().reshape(7,)

                state, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    break

                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.env.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.pth")
    args = parser.parse_args()

    model = Model(args.model)
    model.test()

