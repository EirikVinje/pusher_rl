import os
import optuna
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def objective(trial):

    n_steps = trial.suggest_int("n_steps", 256, 1024)
    gamma = trial.suggest_float("gamma", 0.5, 1.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.5)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    gae_lambda = trial.suggest_float("gae_lambda", 0.5, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 1.0)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)

    vec_env = make_vec_env("Pusher-v4", n_envs=1)
    model = PPO("MlpPolicy", vec_env, 
            verbose=1,
            batch_size=32,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            vf_coef=vf_coef)
    
    model.learn(total_timesteps=100_000,log_interval=10,progress_bar=True)

    res = test(vec_env, model)
    
    return res

def test(vec_env, model):
    state = vec_env.reset()
    tot_reward = 0
    count = 0
    mean_rewards = []
    cum_rewards = []
    for _ in range(1000):
        
        action, _states = model.predict(state)

        state, rewards, dones, info = vec_env.step(action)
        tot_reward += rewards
        count+=1

        if dones:
            cum_rewards.append(tot_reward)
            mean_reward = tot_reward/count
            mean_rewards.append(mean_reward)
            count = 0
            tot_reward = 0
            state = vec_env.reset()
    print(f"Mean rewards: {mean_rewards}")
    print(f"Cumulative rewards: {cum_rewards}")
    print(f"Mean cum reward: {sum(cum_rewards)/len(cum_rewards)}")
    vec_env.close()
    return sum(cum_rewards)/len(cum_rewards)

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial), n_trials=20)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)

    

