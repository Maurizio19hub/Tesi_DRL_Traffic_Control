import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import MyEnv
'''
if __name__ == "__main__":
    save_dir = "models/correct_time/t20energy-pedwaiting_f1"
    os.makedirs(save_dir, exist_ok=True)

    env = MyEnv()
    env = Monitor(env, filename=save_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = 1e-4,
        n_steps       = 3600, 
        batch_size    = 300, # divisore di n_steps => 12
        n_epochs      = 5,
        gamma         = 0.99,
        ent_coef      = 0.05,
        verbose       = 1,
        
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   =  36000,
        save_path   = save_dir,
        name_prefix = "ppo_checkpoint"
    )

    model.learn(total_timesteps=3_000_000, callback=checkpoint_callback)
    model.save(os.path.join(save_dir, "ppo_semaforo"))
    env.save(os.path.join(save_dir,"vec_normalize_stats.pkl"))

    env.close()
    print(f"Training completato — modello salvato in {save_dir}/ppo_semaforo.zip")
'''
if __name__ == "__main__":
    save_dir = "models/correct_time/t20energy-pedwaiting_f1_v2"
    os.makedirs(save_dir, exist_ok=True)
    
    env = MyEnv()
    env = Monitor(env, filename=save_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("models/correct_time/t20energy-pedwaiting_f1/vec_normalize_stats.pkl", env)
    env.training    = True   # continua ad aggiornare le statistiche
    env.norm_reward = True

    model = PPO.load("models/correct_time/t20energy-pedwaiting_f1/ppo_semaforo", env=env)
    model.learning_rate = 3e-4
    model.lr_schedule = lambda _: 3e-4

    checkpoint_callback = CheckpointCallback(
        save_freq   =  36000,
        save_path   = save_dir,
        name_prefix = "ppo_checkpoint"
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    model.save(os.path.join(save_dir, "ppo_semaforo_v2"))
    env.save(os.path.join(save_dir, "vec_normalize_stats_v2.pkl"))