import gymnasium as gym
import argparse
import torch
import os

from ddpg import ActorNetwork


class Model:
    def __init__(self, path_model, render_mode, device):
        
        user = os.environ.get("USER")
        root = f"/home/{user}/data/pusher_models/"
        path_model = os.path.join(root, path_model)

        self.env = gym.make("Pusher-v4", 
                            render_mode=render_mode, 
                            max_episode_steps=200)

        self.device = device

        self.net = ActorNetwork().to(self.device)
        self.net.load_state_dict(torch.load(path_model))

    def test(self, seed=42):
        
        state, info = self.env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        tot_reward = 0
        count = 0
        mean_rewards = []
        cum_rewards = []
        for _ in range(1000):
            
            action = self.net(state)

            if self.device == "cpu":
                action = action.detach().numpy().reshape(7,)
            elif self.device == "cuda":
                action = action.detach().cpu().numpy().reshape(7,)

            state, reward, terminated, truncated, info = self.env.step(action)
            tot_reward += reward
            count+=1

            if terminated or truncated:
                cum_rewards.append(tot_reward)
                mean_reward = tot_reward/count
                mean_rewards.append(mean_reward)
                count = 0
                tot_reward = 0
                state, info = self.env.reset()

            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        print(f"Mean rewards: {mean_rewards}")
        print(f"Cumulative rewards: {cum_rewards}")
        print(f"Mean cum reward: {sum(cum_rewards)/len(cum_rewards)}")
        self.env.close()
        return sum(cum_rewards)/len(cum_rewards)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="run_1/run_1_1000.pt")
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = Model(args.model, args.render_mode, args.device)
    model.test()

