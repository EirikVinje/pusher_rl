import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("Pusher-v4", render_mode='human', max_episode_steps=100)
vec_env = make_vec_env(lambda:env, n_envs=1)

path = os.getcwd()
savepath = os.path.join(path, os.pardir)

model = PPO.load(f"{savepath}/models/ppo_5M_200step_lr")
#model = PPO.load("logs/best_model")
#model = SAC.load(f"{savepath}/models/sac_10M_200ep")

N_SEEDS = 10
seeds = [i for i in range(N_SEEDS)]
succ_rate = 0

rewards_lst=[]
for seed in seeds:
    vec_env.seed(seed)
    state = vec_env.reset()
    rewards_i = []
    while True:
        action, _states = model.predict(state)
        state, rewards, dones, info = vec_env.step(action)
        #print(rewards)
        #vec_env.render("human")
        rewards_i.append(rewards[0])
        


        if dones:
            break
    rewards_lst.append(rewards_i)
    if rewards_i[-1]>=-0.12:
        succ_rate += 1
    print(f"Seed: {seed}, Reward on last step: {rewards_i[-1]:.4f}")

vec_env.close()

print(f"Success rate: {succ_rate/N_SEEDS}")
