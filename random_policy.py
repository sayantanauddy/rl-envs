import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import rl_envs  

VIDEO_DIR = "./videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

def make_env():
    env = gym.make("RampPush-v0", render_mode="rgb_array")
    env = Monitor(env)
    # Record every episode
    env = RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda e: True)
    return env

def run_random_policy(n_episodes=3):
    env = make_env()
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        print(f"Episode {ep+1}: steps={step}, total_reward={total_reward:.2f}")
    env.close()
    print(f"Videos saved to {VIDEO_DIR}")

if __name__ == "__main__":
    run_random_policy()
