from gymnasium.envs.registration import register

register(
    id="RampPush-v0",
    entry_point="rl_envs.envs:RampPushEnv",
    max_episode_steps=500,
    )
