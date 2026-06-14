import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traci
import subprocess
import os
from sumo_env import MyEnv

# ── Configurazione ────────────────────────────────────────────
SEEDS      = list(range(0, 100, 5))   # stessi seed del PPO
OUTPUT_DIR = "results/baseline_optimal-phase-f2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SPEED = 13.89
BRANCHES  = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}

SUMO_CFG = "simulazione.sumocfg"
TLS_ID   = "tls_1"

# ── Struttura per raccogliere metriche per episodio ───────────
all_episodes = []

for seed in SEEDS:
    print(f"\nEpisodio seed={seed}...")

    # Genera pedoni con stesso seed del PPO
    subprocess.run([
        "python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
        "-n", "incrocio3_300mf.net.xml",
        "-e", "3600",
        "-p", "3.0",
        "--persontrips",
        "-r", "pedoni.rou.xml",
        "--seed", str(seed)
    ], check=True, stdout=subprocess.DEVNULL)

    sumo_cmd = [
        "sumo", "-c", SUMO_CFG,
        "--seed",               str(seed),
        "--waiting-time-memory","1000",
        "--no-step-log",        "true",
        "--no-warnings",        "true",
        "--time-to-teleport",   "100",
        "--collision.action",   "teleport",
        "--collision.mingap-factor", "0",
        "--collision.check-junctions", "true",
    ]

    if traci.isLoaded():
        traci.close()
    traci.start(sumo_cmd)

    # Metriche step-by-step
    waiting_times    = []
    queue_lengths    = []
    queue_max        = []
    speeds           = []
    arrived_per_step = []
    ped_waiting      = []
    co2_per_step     = []
    nox_per_step     = []
    pmx_per_step     = []
    fuel_per_step    = []
    collisions_list  = []
    teleports_list   = []
    phase_changes    = 0
    prev_phase       = None
    total_reward     = 0.0

    co2_per_step_val = 0.0
    nox_per_step_val = 0.0
    pmx_per_step_val = 0.0
    fuel_per_step_val = 0.0

    for step in range(3600):
        # Libera evoluzione — SUMO gestisce il semaforo autonomamente
        traci.simulationStep()

        # ── Metriche da TraCI ─────────────────────────────────
        w_sum, q_sum, q_max_val, s_sum, s_count = 0, 0, 0, 0, 0

        for branch_name, (edge, num_lanes) in BRANCHES.items():
            veh_ids = traci.edge.getLastStepVehicleIDs(edge)
            n_veh   = len(veh_ids)
            halting = traci.edge.getLastStepHaltingNumber(edge)

            w_sum += traci.edge.getWaitingTime(edge)

            q_m = halting * 5
            q_sum    += q_m
            q_max_val = max(q_max_val, q_m)

            if n_veh > 0:
                s_sum   += traci.edge.getLastStepMeanSpeed(edge) * n_veh
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

        ped_w = sum(
            1 for pid in traci.person.getIDList()
            if traci.person.getWaitingTime(pid) > 0
        )
        ped_waiting.append(ped_w)

        collisions_list.append(len(traci.simulation.getCollisions()))
        teleports_list.append(traci.simulation.getStartingTeleportNumber())

        current_phase = traci.trafficlight.getPhase(TLS_ID)
        if prev_phase is not None and current_phase != prev_phase:
            phase_changes += 1
        prev_phase = current_phase

    traci.close()

    all_episodes.append({
        "seed":             seed,
        "avg_waiting_time": np.mean(waiting_times),
        "avg_queue_length": np.mean(queue_lengths),
        "max_queue_length": np.max(queue_max),
        "avg_speed":        np.mean(speeds),
        "total_arrived":    np.sum(arrived_per_step),
        "phase_changes":    phase_changes,
        "avg_ped_waiting":  np.mean(ped_waiting),
        "total_co2":        np.sum(co2_per_step),
        "total_nox":        np.sum(nox_per_step),
        "total_pmx":        np.sum(pmx_per_step),
        "total_fuel":       np.sum(fuel_per_step),
        "total_collisions": np.sum(collisions_list),
        "total_teleports":  np.sum(teleports_list),
        "_waiting_times":   waiting_times,
        "_queue_lengths":   queue_lengths,
        "_speeds":          speeds,
        "_co2":             co2_per_step,
        "_collisions":      collisions_list,
    })

    print(f"  arrived={np.sum(arrived_per_step)} | "
          f"collisions={np.sum(collisions_list)} | phase_changes={phase_changes}")

# ── DataFrame riepilogativo ───────────────────────────────────
df = pd.DataFrame([{k: v for k, v in ep.items() if not k.startswith('_')}
                   for ep in all_episodes])
df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
print(f"\nMetriche salvate in {OUTPUT_DIR}/metrics.csv")
print(df.describe().round(2))

# ── GRAFICI (identici al PPO ma con colore diverso) ───────────
C_BASE = "#4CAF50"
C_MEAN = "#FF5722"

def make_timeseries(key, label, ylabel, color=C_BASE, ax=None):
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

# Figura 1: Serie temporali
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Baseline — Serie Temporali (media ± std su 20 episodi)", fontsize=14)

make_timeseries("_waiting_times", "Tempo medio di attesa veicoli (s)", "secondi", ax=axes[0,0])
make_timeseries("_queue_lengths", "Lunghezza media code (m)",          "metri",   ax=axes[0,1])
make_timeseries("_speeds",        "Velocità media veicoli (m/s)",      "m/s",     ax=axes[1,0])
make_timeseries("_co2",           "Emissioni CO₂ per step (mg/s)",     "mg/s",    ax=axes[1,1])
make_timeseries("_collisions",    "Collisioni per step",               "n°",      ax=axes[2,0])

arrived_series = np.array([np.cumsum(ep["_waiting_times"]) for ep in all_episodes])
mean_arr = arrived_series.mean(axis=0)
axes[2,1].plot(mean_arr, color=C_BASE, linewidth=1.5)
axes[2,1].set_title("Cumulativo — proxy attesa totale")
axes[2,1].set_xlabel("Step simulazione")
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/timeseries.png", dpi=150, bbox_inches="tight")
print("Salvato: timeseries.png")

# Figura 2: Boxplot
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Baseline — Distribuzione Metriche per Episodio", fontsize=14)

metrics_box = [
    ("avg_waiting_time", "Attesa media\nveicoli (s)"),
    ("avg_queue_length", "Coda media (m)"),
    ("max_queue_length", "Coda massima (m)"),
    ("avg_speed",        "Velocità media\n(m/s)"),
    ("total_arrived",    "Veicoli arrivati"),
    ("phase_changes",    "Cambi fase\nsemaforica"),
    ("total_collisions", "Collisioni totali"),
    ("avg_ped_waiting",  "Pedoni in attesa\n(media)"),
]

for ax, (col, label) in zip(axes.flat, metrics_box):
    ax.boxplot(df[col], patch_artist=True,
               boxprops=dict(facecolor=C_BASE, alpha=0.6),
               medianprops=dict(color=C_MEAN, linewidth=2))
    ax.set_title(label, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/boxplot.png", dpi=150, bbox_inches="tight")
print("Salvato: boxplot.png")

# Figura 3: Emissioni
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Baseline — Emissioni e Consumo Energetico", fontsize=13)

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

plt.show()
print("\nDone.")