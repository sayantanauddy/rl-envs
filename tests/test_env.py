import numpy as np
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt 
import gymnasium as gym
import rl_envs

def _env(env):

    print(env.action_space)
    print(env.observation_space)

    # SB3 check_env
    print("Running check_env...")
    check_env(env, warn=True)
    print("check_env passed!")

    # Single rollout
    obs, _ = env.reset()
    done = False
    step_count = 0
    print("Running a single rollout...")
    while not done and step_count < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Check reward type
        assert isinstance(reward, float), f"Reward is not float! Got {type(reward)}"
        step_count += 1

    print(f"Rollout finished after {step_count} steps. Rewards and obs OK.")

    # Rendering
    if hasattr(env, "render_mode"):
        #env.render_mode = "rgb_array"
        img = env.render()
        assert isinstance(img, np.ndarray), "Render did not return an image"
        print("Render OK. Image shape:", img.shape)
        plt.imshow(img)

    env.close()
    print("Environment check completed successfully!")


def test_envs():
    
    env = gym.make("RampPush-v0", render_mode="rgb_array", width=640, height=480)
    _env(env)

if __name__ == "__main__":
    
    test_envs()
