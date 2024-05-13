import gymnasium as gym
import argparse
import torch
import os

from dqn import ActorNetwork


class Model:
    def __init__(self, path_model, device="cuda"):
        
        user = os.environ.get("USER")
        root = f"/home/{user}/data/pusher_models/"
        path_model = os.path.join(root, path_model)

        self.env = gym.make("Pusher-v4", 
                            render_mode="human", 
                            max_episode_steps=200)

        self.device = device

        self.net = ActorNetwork().to(self.device)
        self.net.load_state_dict(torch.load(path_model))

    def test(self, seed=42):
        
        state, info = self.env.reset(seed=seed)

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        for _ in range(1000):
            
            action = self.net(state)

            if self.device == "cpu":
                action = action.detach().numpy().reshape(7,)
            elif self.device == "cuda":
                action = action.detach().cpu().numpy().reshape(7,)

            state, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                state, info = self.env.reset(seed=42)

            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.env.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.pth")
    args = parser.parse_args()

    model = Model(args.model)
    model.test()

