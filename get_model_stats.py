import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_env import MyEnv
import os

# ── Configurazione ────────────────────────────────────────────
MODEL_PATH  = "models/correct_time/t20energy-pedwaiting-collisions_f2/ppo_semaforo"
STATS_PATH  = "models/correct_time/t20energy-pedwaiting-collisions_f2/vec_normalize_stats.pkl"
SEEDS       = list(range(0, 100, 5))   # 20 seed fissi
OUTPUT_DIR  = "results/ppo_pedwaiting-collisions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SPEED   = 13.89  # m/s
BRANCHES    = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}

# ── Carica modello ────────────────────────────────────────────
model = PPO.load(MODEL_PATH)
env   = DummyVecEnv([lambda: Monitor(MyEnv())])
env   = VecNormalize.load(STATS_PATH, env)
env.training    = False
env.norm_reward = True

# ── Struttura per raccogliere metriche per episodio ───────────
all_episodes = []

for seed in SEEDS:
    print(f"\nEpisodio seed={seed}...")

    env.env_method("reset", seed=seed)
    obs = env.reset()

    # Metriche step-by-step
    waiting_times     = []   # tempo attesa medio veicoli
    queue_lengths     = []   # lunghezza media code (m)
    queue_max         = []   # lunghezza massima coda (m)
    speeds            = []   # velocità media veicoli
    arrived_per_step  = []   # veicoli arrivati per step
    ped_waiting       = []   # pedoni in attesa
    co2_per_step      = []   # emissioni CO2
    nox_per_step      = []   # emissioni NOx
    pmx_per_step      = []   # emissioni PMx
    fuel_per_step     = []   # consumo carburante
    collisions_list   = []   # collisioni per step
    teleports_list    = []   # teleport per step
    phase_changes     = 0
    action_counts     = {0: 0, 1: 0}
    prev_phase        = None
    total_reward      = 0.0

    co2_per_step_val = 0.0
    nox_per_step_val = 0.0
    pmx_per_step_val = 0.0
    fuel_per_step_val = 0.0


    for step in range(3600):
        action, _ = model.predict(obs, deterministic=True)
        action_counts[int(action[0])] += 1
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        # ── Metriche da TraCI ─────────────────────────────────
        w_sum, q_sum, q_max_val, s_sum, s_count = 0, 0, 0, 0, 0

        for branch_name, (edge, num_lanes) in BRANCHES.items():
            veh_ids = traci.edge.getLastStepVehicleIDs(edge)
            n_veh   = len(veh_ids)
            halting = traci.edge.getLastStepHaltingNumber(edge)
            length  = traci.lane.getLength(edge + "_0")

            # Attesa
            w_sum += traci.edge.getWaitingTime(edge)

            # Code
            q_m = halting * 5  # metri
            q_sum += q_m
            q_max_val = max(q_max_val, q_m)

            # Velocità
            if n_veh > 0:
                s_sum += traci.edge.getLastStepMeanSpeed(edge) * n_veh
                s_count += n_veh

            # Emissioni e carburante
            for veh_id in veh_ids:
                co2_per_step_val += traci.vehicle.getCO2Emission(veh_id)
                nox_per_step_val += traci.vehicle.getNOxEmission(veh_id)
                pmx_per_step_val += traci.vehicle.getPMxEmission(veh_id)
                fuel_per_step_val += traci.vehicle.getFuelConsumption(veh_id)

        co2_per_step.append(co2_per_step_val)
        nox_per_step.append(nox_per_step_val)
        pmx_per_step.append(pmx_per_step_val)
        fuel_per_step.append(fuel_per_step_val)

        waiting_times.append(w_sum / max(s_count, 1))
        queue_lengths.append(q_sum / 4)
        queue_max.append(q_max_val)
        speeds.append(s_sum / max(s_count, 1))
        arrived_per_step.append(traci.simulation.getArrivedNumber())

        # Pedoni
        ped_w = sum(
            1 for pid in traci.person.getIDList()
            if traci.person.getWaitingTime(pid) > 0
        )
        ped_waiting.append(ped_w)

        # Collisioni e teleport
        collisions_list.append(len(traci.simulation.getCollisions()))
        teleports_list.append(traci.simulation.getStartingTeleportNumber())

        # Cambi fase
        current_phase = traci.trafficlight.getPhase("tls_1")
        if prev_phase is not None and current_phase != prev_phase:
            phase_changes += 1
        prev_phase = current_phase

        if done[0]:
            break

    # ── Aggregazione per episodio ─────────────────────────────
    all_episodes.append({
        "seed":                seed,
        "total_reward":        total_reward,
        "avg_waiting_time":    np.mean(waiting_times),
        "avg_queue_length":    np.mean(queue_lengths),
        "max_queue_length":    np.max(queue_max),
        "avg_speed":           np.mean(speeds),
        "total_arrived":       np.sum(arrived_per_step),
        "phase_changes":       phase_changes,
        "avg_ped_waiting":     np.mean(ped_waiting),
        "total_co2":           np.sum(co2_per_step),
        "total_nox":           np.sum(nox_per_step),
        "total_pmx":           np.sum(pmx_per_step),
        "total_fuel":          np.sum(fuel_per_step),
        "total_collisions":    np.sum(collisions_list),
        "total_teleports":     np.sum(teleports_list),
        "action_0_pct":        action_counts[0] / 3600,
        "action_1_pct":        action_counts[1] / 3600,
        # Serie temporali per grafici dettagliati
        "_waiting_times":      waiting_times,
        "_queue_lengths":      queue_lengths,
        "_speeds":             speeds,
        "_co2":                co2_per_step,
        "_collisions":         collisions_list,
    })

    print(f"  reward={total_reward:.1f} | arrived={np.sum(arrived_per_step)} | "
          f"collisions={np.sum(collisions_list)} | phase_changes={phase_changes}")

env.close()

# ── DataFrame riepilogativo ───────────────────────────────────
df = pd.DataFrame([{k: v for k, v in ep.items() if not k.startswith('_')}
                   for ep in all_episodes])
df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
print(f"\nMetriche salvate in {OUTPUT_DIR}/metrics.csv")
print(df.describe().round(2))

# ── GRAFICI ───────────────────────────────────────────────────

C_PPO  = "#2196F3"
C_MEAN = "#FF5722"

def make_timeseries(key, label, ylabel, color=C_PPO, ax=None):
    """Traccia la serie temporale media con banda di confidenza."""
    series = np.array([ep[key] for ep in all_episodes])
    mean   = series.mean(axis=0)
    std    = series.std(axis=0)
    steps  = np.arange(len(mean))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(steps, mean-std, mean+std, alpha=0.2, color=color)
    ax.set_xlabel("Step simulazione")
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    return ax

# ── Figura 1: Serie temporali principali ─────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("PPO — Serie Temporali (media ± std su 20 episodi)", fontsize=14)

make_timeseries("_waiting_times", "Tempo medio di attesa veicoli (s)",  "secondi",  ax=axes[0,0])
make_timeseries("_queue_lengths", "Lunghezza media code (m)",           "metri",    ax=axes[0,1])
make_timeseries("_speeds",        "Velocità media veicoli (m/s)",       "m/s",      ax=axes[1,0])
make_timeseries("_co2",           "Emissioni CO₂ per step (mg/s)",      "mg/s",     ax=axes[1,1])
make_timeseries("_collisions",    "Collisioni per step",                "n°",       ax=axes[2,0])

# Veicoli arrivati cumulati
arrived_series = np.array([np.cumsum(ep["_waiting_times"]) for ep in all_episodes])
mean_arr = arrived_series.mean(axis=0)
axes[2,1].plot(mean_arr, color=C_PPO, linewidth=1.5)
axes[2,1].set_title("Cumulativo — proxy attesa totale")
axes[2,1].set_xlabel("Step simulazione")
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/timeseries.png", dpi=150, bbox_inches="tight")
print("Salvato: timeseries.png")

# ── Figura 2: Metriche aggregate (boxplot) ────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("PPO — Distribuzione Metriche per Episodio", fontsize=14)

metrics_box = [
    ("avg_waiting_time",  "Attesa media\nveicoli (s)"),
    ("avg_queue_length",  "Coda media (m)"),
    ("max_queue_length",  "Coda massima (m)"),
    ("avg_speed",         "Velocità media\n(m/s)"),
    ("total_arrived",     "Veicoli arrivati"),
    ("phase_changes",     "Cambi fase\nsemaforica"),
    ("total_collisions",  "Collisioni totali"),
    ("avg_ped_waiting",   "Pedoni in attesa\n(media)"),
]

for ax, (col, label) in zip(axes.flat, metrics_box):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor=C_PPO, alpha=0.6),
               medianprops=dict(color=C_MEAN, linewidth=2))
    ax.set_title(label, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/boxplot.png", dpi=150, bbox_inches="tight")
print("Salvato: boxplot.png")

# ── Figura 3: Emissioni e energia ─────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("PPO — Emissioni e Consumo Energetico", fontsize=13)

emissioni = [
    ("total_co2",  "CO₂ totale (mg)", "#E53935"),
    ("total_nox",  "NOx totale (mg)", "#FB8C00"),
    ("total_pmx",  "PMx totale (mg)", "#8E24AA"),
    ("total_fuel", "Carburante (ml)", "#43A047"),
]

for ax, (col, label, color) in zip(axes, emissioni):
    ax.bar(range(len(df)), df[col], color=color, alpha=0.7)
    ax.axhline(df[col].mean(), color='black', linestyle='--', linewidth=1.5,
               label=f"Media: {df[col].mean():.1f}")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Episodio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/emissions.png", dpi=150, bbox_inches="tight")
print("Salvato: emissions.png")

# ── Figura 4: Distribuzione azioni ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Mantieni (0)", "Cambia (1)"],
       [df["action_0_pct"].mean(), df["action_1_pct"].mean()],
       color=[C_PPO, C_MEAN], alpha=0.8)
ax.set_ylabel("Frazione media")
ax.set_title("Distribuzione azioni agente PPO")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/actions.png", dpi=150, bbox_inches="tight")
print("Salvato: actions.png")

plt.show()
print("\nDone.")