import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import MyEnv

import csv
from stable_baselines3.common.callbacks import BaseCallback

class PhysicalMetricsCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_count = 0
        self.csv_file = os.path.join(save_path, 'training_physical_metrics.csv')
        self._init_csv()

    def _init_csv(self):
        fieldnames = [
            'episode', 'training_timestep', 'episode_reward',
            'mean_queue', 'max_queue', 'mean_waiting_time', 'mean_speed',
            'total_CO2', 'total_NOx', 'total_PMx', 'total_fuel',
            'arrived_vehicles', 'pedestrian_waiting', 'phase_changes',
            'collisions', 'teleports'
        ]
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if infos and infos[0].get('episode') is not None:
            ep_info = infos[0]['episode']
            episode_reward = ep_info['r']

            physical = infos[0].get('physical_metrics')

            row = {
                'episode': self.episode_count,
                'training_timestep': self.num_timesteps,
                'episode_reward': episode_reward,
                **physical
            }
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
            self.episode_count += 1
        return True

if __name__ == "__main__":
    save_dir = "models/t20energy-pedwaiting_f3"
    os.makedirs(save_dir, exist_ok=True)

    env = MyEnv()
    env = Monitor(env, filename=save_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = 3e-4,
        n_steps       = 3600, 
        batch_size    = 300, # divisore di n_steps => 12
        n_epochs      = 5,
        gamma         = 0.99,
        ent_coef      = 0.01,
        verbose       = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = 3600,
        save_path   = save_dir,
        name_prefix = "ppo_checkpoint"
    )

    physical_callback = PhysicalMetricsCallback(save_path=save_dir)

    model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, physical_callback])
    model.save(os.path.join(save_dir, "ppo_semaforo"))
    env.save(os.path.join(save_dir,"vec_normalize_stats.pkl"))

    env.close()
    print(f"Training completato — modello salvato in {save_dir}/ppo_semaforo.zip")