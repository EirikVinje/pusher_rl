# Usage code
import gymnasium as gym
import renderlab as rl
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

repo_id = "VinayHajare/ppo-Pusher-v4"
filename = "ppo-Pusher-v4.zip"

eval_env = gym.make("Pusher-v4",render_mode="human")
checkpoint = load_from_hub(repo_id, filename)
model = PPO.load(checkpoint,env=eval_env,print_system_info=True)

mean_reward, std_reward = evaluate_policy(model,eval_env, n_eval_episodes=5, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Enjoy trained agent
env = eval_env
env = rl.RenderFrame(env,"./output")
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, rewards, terminated, truncated, info = env.step(action)
env.play()
