import gymnasium as gym


def main():
        
    env = gym.make("Pusher-v4", render_mode="human")

    env.action_space.seed(42)
    
    observation, info = env.reset(seed=42)

    for _ in range(10000):
        
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=42)

    env.close()

if __name__ == "__main__":
    main()