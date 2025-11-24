import os
import time
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# So that gym.make("RampPush-v0") works
import rl_envs  


def make_env(env_id, monitor_dir, rank=0, render_mode=None, seed=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        if seed is not None:
            try:
                env.reset(seed=seed + rank)
            except TypeError:
                pass
        os.makedirs(monitor_dir, exist_ok=True)
        mon_file = os.path.join(monitor_dir, f'monitor_{rank}.csv')
        env = Monitor(env, filename=mon_file)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="RampPush-v0")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--algo", type=str, default="SAC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    base_log_dir = os.path.join(".logs", f"{args.algo}_{args.env}", run_id)
    models_dir = os.path.join(base_log_dir, "models")
    video_dir = os.path.join(base_log_dir, "videos")
    monitor_dir = os.path.join(base_log_dir, "monitors")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.env, monitor_dir, rank=0, render_mode=None, seed=args.seed)])

    eval_env_raw = gym.make(args.env, render_mode="rgb_array")
    eval_env = Monitor(
        gym.wrappers.RecordVideo(
            eval_env_raw,
            video_folder=video_dir,
            episode_trigger=lambda e: True
        ),
        filename=os.path.join(monitor_dir, "eval_monitor.csv")
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(base_log_dir, "tensorboard"),
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=models_dir, name_prefix="sac_model")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=os.path.join(base_log_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    final_path = os.path.join(models_dir, "sac_final")
    model.save(final_path)
    print(f"Training finished. Model saved to: {final_path}")
    print(f"Videos saved in: {video_dir}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
