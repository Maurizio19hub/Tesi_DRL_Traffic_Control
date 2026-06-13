from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_env import MyEnv

model = PPO.load("models/correct_time/t20energy-pedwaiting-collisions_f2/ppo_semaforo")

env = DummyVecEnv([lambda: Monitor(MyEnv())])
env = VecNormalize.load("models/correct_time/t20energy-pedwaiting-collisions_f2/vec_normalize_stats.pkl", env)
env.training    = False
env.norm_reward = True

mean_rew = 0.0
action_counts = {0: 0, 1: 0}
for ep in range(1):
    obs = env.reset()
    total = 0
    for i in range(3600):
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action[0])] += 1
        obs, reward, done, info = env.step(action)
        total += reward[0]
        if done[0]:
            break
    mean_rew += total
    print(f"Episodio {ep+1}: {total:.2f}")

print(f"\nPPO Mean reward: {mean_rew/10:.2f}")
print(f"Action 0: {action_counts[0]} ({action_counts[0]/3600:.1%})")
print(f"Action 1: {action_counts[1]} ({action_counts[1]/3600:.1%})")
env.close()