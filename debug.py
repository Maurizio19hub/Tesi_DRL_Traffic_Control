from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_env import MyEnv

model = PPO.load("models/correct_time/t15_f1/ppo_semaforo")

env = DummyVecEnv([lambda: Monitor(MyEnv())])
env = VecNormalize.load("models/correct_time/t15_f1/vec_normalize_stats.pkl", env)
env.training    = False
env.norm_reward = False

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True
)
print(f"PPO Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
env.close()