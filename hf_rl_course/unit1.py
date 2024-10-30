from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Understanding Gymnasium and how it works
env = gym.make("LunarLander-v2")
observation, info = env.reset()
for _ in range(20):
    action = env.action_space.sample()      # Take a random action
    # Do this action in the environment and get next_state, reward, terminated, truncated and info
    observation, reward, terminated, truncated, info = env.step(action)
    print("Action taken:", action, "\tReward:", reward)
    # If the game is terminated (in our case we land, crashed) or truncated (timeout)
    if terminated or truncated:
        print("Environment is reset")
        observation, info = env.reset()     # Reset the environment
env.close()


# Training PPO on LunarLander-v2
env = make_vec_env('LunarLander-v2', n_envs=16)
model = PPO(policy='MlpPolicy', env=env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01, verbose=1)
model.learn(total_timesteps=1000000)
model_name = "ppo-LunarLander-v2"
model.save(model_name)


# Evaluate the agent
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

