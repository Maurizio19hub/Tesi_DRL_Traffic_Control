from stable_baselines3 import PPO
from sumo_env import MyEnv

model = PPO.load("ppo_semaforo")
env = MyEnv()

obs, info = env.reset()
for _ in range(10800):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.close()