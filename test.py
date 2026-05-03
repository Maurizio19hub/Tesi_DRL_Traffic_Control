# test_env.py
from sumo_env import MyEnv
import numpy as np

env = MyEnv()

# Test reset
print("=== TEST RESET ===")
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Obs values: {obs}")
print(f"Obs min: {obs.min():.3f}, max: {obs.max():.3f}")

# Verifica bounds
assert obs.shape == (15,), f"Shape sbagliata: {obs.shape}"
assert obs.min() >= 0.0, "Valore sotto 0"
assert obs.max() <= 1.0, "Valore sopra 1"
print("✅ Bounds OK")

# Test step con azione 0
print("\n=== TEST STEP azione=0 ===")
obs, reward, terminated, truncated, info = env.step(0)
print(f"Obs: {obs}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Info: {info}")

# Test step con azione 1
for i in range(1, 20):
    print(f"\n=== TEST STEP azione={i} ===")
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"Obs: {obs}")
    print(f"Info: {info}")

env.close()
print("\n✅ Test completato")