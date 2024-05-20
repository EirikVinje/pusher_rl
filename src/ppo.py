import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("Pusher-v4", n_envs=1)

'''model = PPO("MlpPolicy", vec_env, 
            verbose=1,
            batch_size=32,
            n_steps=512,
            gamma=0.9,
            learning_rate=0.0001,
            ent_coef=7.52585e-08,
            clip_range=0.3,
            n_epochs=5,
            gae_lambda=1.0,
            max_grad_norm=0.9,
            vf_coef=0.950368)'''

#model.learn(total_timesteps=1_000_000,log_interval=10,progress_bar=True)
#model.save("ppo_cartpole_1M")

#del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole_1M")


seeds = [3559, 3216, 7890, 5242, 4924, 3588, 722, 8119]


for seed in seeds:
    vec_env.seed(seed)
    state = vec_env.reset()
    while True:
        action, _states = model.predict(state)
        state, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

        if dones:
            break