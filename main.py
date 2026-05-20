import os
from stable_baselines3 import PPO
from sumo_env import MyEnv

if __name__ == "__main__":
    save_dir = "models/t5"
    os.makedirs(save_dir, exist_ok=True)

    env = MyEnv()

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        learning_rate = 1e-4, # test precedente con 3e-4
        n_steps       = 2160, # 10800/DECISION_INTERVAL
        batch_size    = 216, # divisore di n_steps => 10
        n_epochs      = 10,
        gamma         = 0.99,
        ent_coef      = 0.01,
        verbose       = 1,
    )

    model.learn(total_timesteps=500_000)
    model.save(os.path.join(save_dir, "ppo_semaforo"))

    env.close()
    print(f"Training completato — modello salvato in {save_dir}/ppo_semaforo.zip")