import numpy as np
from sumo_env import MyEnv
import traci

class FixedPhaseEnv(MyEnv):
    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.last_cost = 0
        traci.trafficlight.setProgram(self.tls_id, "0")
        return result

    def step(self, action):
        traci.simulationStep()
        self.current_step += 1
        sumo_phase = traci.trafficlight.getPhase(self.tls_id)
        self.current_phase_index = 0 if sumo_phase in [0, 1] else 1

        if self.current_step >= self.max_steps:
            traci.close()
            return np.zeros(19, dtype=np.float32), 0.0, True, False, {}

        obs        = self._get_observation()
        reward     = self._get_cost()
        terminated = self.current_step >= self.max_steps
        if terminated:
            traci.close()
        return obs, reward, terminated, False, {}


SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS = [1]
all_rewards = []
env = FixedPhaseEnv()

for seed in SEEDS:
    obs, info = env.reset(seed=seed)
    ep_reward = 0.0
    for _ in range(3600):
        obs, reward, terminated, truncated, info = env.step(0)
        ep_reward += reward
        if terminated:
            break
    all_rewards.append(ep_reward)
    print(f"Seed={seed}: reward={ep_reward:.2f}")

env.close()
print(f"\nBaseline: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")