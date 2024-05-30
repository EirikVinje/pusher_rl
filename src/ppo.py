import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Parallel environments
env = gym.make("Pusher-v4", render_mode='rgb_array', max_episode_steps=200)
vec_env = make_vec_env(lambda:env, n_envs=1)

path = os.getcwd()
# get parent directory
savepath = os.path.join(path, os.pardir)

#params = {'n_steps': 878, 'gamma': 0.8680218654215721, 'learning_rate': 0.00017445135385783767, 'ent_coef': 0.0011388540424663318, 'clip_range': 0.1964158980973974, 'n_epochs': 8, 'gae_lambda': 0.8124992408081275, 'max_grad_norm': 0.9961750824042535, 'vf_coef': 0.7726884965404387}
#params={'n_steps': 483, 'gamma': 0.9733319984142623, 'learning_rate': 6.249837257325398e-05, 'ent_coef': 6.2177665117053e-05, 'clip_range': 0.20674514787972498, 'n_epochs': 18, 'gae_lambda': 0.9180827852846369, 'max_grad_norm': 0.7932345709898225, 'vf_coef': 0.5888144289366819}

'''
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
            tensorboard_log="./ppo_tensorboard_log/")'''

eval_callback = EvalCallback(vec_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

model = PPO("MlpPolicy", vec_env, 
            verbose=1,
            learning_rate=0.0001,
            clip_range=0.2,
            tensorboard_log="./ppo_tensorboard_log/")

model.learn(total_timesteps=5_000_000,log_interval=10,progress_bar=True, callback=eval_callback)
#model.save(f"{savepath}/models/ppo_5M_200step_standard_v2")

#del model # remove to demonstrate saving and loading

#model = PPO.load(f"{savepath}/models/ppo_200K_150ep")

'''
seeds = [3559, 3216, 7890, 5242, 4924, 3588, 722, 8119]


for seed in seeds:
    vec_env.seed(seed)
    state = vec_env.reset()
    while True:
        action, _states = model.predict(state)
        state, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")

        if dones:
            break'''