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

params = {'n_steps': 878, 'gamma': 0.8680218654215721, 'learning_rate': 0.00017445135385783767, 'ent_coef': 0.0011388540424663318, 'clip_range': 0.1964158980973974, 'n_epochs': 8, 'gae_lambda': 0.8124992408081275, 'max_grad_norm': 0.9961750824042535, 'vf_coef': 0.7726884965404387}

model = PPO("MlpPolicy", vec_env, 
            verbose=1,
            batch_size=32,
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            learning_rate=params['learning_rate'],
            ent_coef=params['ent_coef'],
            clip_range=params['clip_range'],
            n_epochs=params['n_epochs'],
            gae_lambda=params['gae_lambda'],
            max_grad_norm=params['max_grad_norm'],
            vf_coef=params['vf_coef'],
            tensorboard_log="./ppo_tensorboard_log/")

model.learn(total_timesteps=1_000_000,log_interval=10,progress_bar=True)
model.save("ppo_cartpole_1M")

#del model # remove to demonstrate saving and loading

#model = PPO.load("ppo_cartpole_1M")


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