from stable_baselines3 import PPO
from sumo_env import MyEnv

model = PPO.load("models/t11/ppo_semaforo")
env   = MyEnv()
obs, _ = env.reset()

action_counts = {0: 0, 1: 0}
phase_changes = 0
total_reward = 0.0
step = 0

for _ in range(3600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    action_counts[int(action)] += 1
    if action == 1:
        phase_changes += 1

    if terminated:
        break

env.close()

print(f"\n=== RISULTATI ===")
print(f"Action 0 (mantieni): {action_counts[0]} volte ({action_counts[0]/sum(action_counts.values()):.1%})")
print(f"Action 1 (cambia):   {action_counts[1]} volte ({action_counts[1]/sum(action_counts.values()):.1%})")
print(f"Cambi di fase effettivi (rispettando min_green): da vedere in sumo-gui")
print(f"  Total reward:           {total_reward:.2f}")
print(f"  Step:                   {step}")
print(f"  Reward media per step:  {total_reward / step:.5f}")