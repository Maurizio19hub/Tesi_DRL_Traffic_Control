"""
generate_csv.py
─────────────────────────────────────────────────────────────
Genera il file CSV con le metriche fisiche raccolte durante
la valutazione, sia per la baseline (fasi fisse, libera
evoluzione) sia per il modello PPO addestrato.
─────────────────────────────────────────────────────────────
"""
 
import os
import subprocess
import numpy as np
import pandas as pd
import traci
 
# ── Import condizionale (solo se serve generare il PPO) ───────
def _lazy_import_ppo():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    return PPO, Monitor, DummyVecEnv, VecNormalize
 

 
# ── Configurazione comune ──────────────────────────────────────
SEEDS      = list(range(0, 100, 5))   # 20 seed fissi, usati sia da baseline che da PPO
MAX_STEPS  = 3600
TLS_ID     = "tls_1"
NET_FILE   = "incrocio3_300mf.net.xml"
SUMO_CFG   = "simulazione.sumocfg"
 
BRANCHES = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}
 
 
# ────────────────────────────────────────────────────────────────
# Funzioni di raccolta metriche per singolo step (comuni)
# ────────────────────────────────────────────────────────────────
def collect_step_metrics(prev_phase, phase_changes):
    """
    Legge da TraCI tutte le metriche fisiche relative all'ultimo
    step di simulazione e ritorna un dizionario con i valori
    istantanei (non cumulativi) per quello step.
    """
    w_sum, q_sum, q_max_val, s_sum, s_count = 0.0, 0.0, 0.0, 0.0, 0.0
    co2_step, nox_step, pmx_step, fuel_step = 0.0, 0.0, 0.0, 0.0
 
    for branch_name, (edge, num_lanes) in BRANCHES.items():
        veh_ids = traci.edge.getLastStepVehicleIDs(edge)
        n_veh   = len(veh_ids)
        halting = traci.edge.getLastStepHaltingNumber(edge)
 
        w_sum += traci.edge.getWaitingTime(edge)
 
        q_m = halting * 5  # lunghezza media veicolo in metri
        q_sum += q_m
        q_max_val = max(q_max_val, q_m)
 
        if n_veh > 0:
            s_sum   += traci.edge.getLastStepMeanSpeed(edge) * n_veh
            s_count += n_veh
 
        # ── Emissioni istantanee per questo step ──────────────
        # getCO2Emission/getNOxEmission/getPMxEmission/getFuelConsumption
        # restituiscono il valore in mg/s relativo
        # all'ultimo step di simulazione.
        # Con step-length = 1s questo è già il contributo dello step.
        for veh_id in veh_ids:
            co2_step  += traci.vehicle.getCO2Emission(veh_id)
            nox_step  += traci.vehicle.getNOxEmission(veh_id)
            pmx_step  += traci.vehicle.getPMxEmission(veh_id)
            fuel_step += traci.vehicle.getFuelConsumption(veh_id)
 
    ped_waiting = sum(
        1 for pid in traci.person.getIDList()
        if traci.person.getWaitingTime(pid) > 0
    )
 
    n_collisions = len(traci.simulation.getCollisions())
    n_teleports  = traci.simulation.getStartingTeleportNumber()
    n_arrived    = traci.simulation.getArrivedNumber()
 
    current_phase = traci.trafficlight.getPhase(TLS_ID)
    if prev_phase is not None and current_phase != prev_phase:
        phase_changes += 1
 
    return {
        "waiting":     w_sum / max(s_count, 1),
        "queue_len":   q_sum / 4,
        "queue_max":   q_max_val,
        "speed":       s_sum / max(s_count, 1),
        "arrived":     n_arrived,
        "ped_waiting": ped_waiting,
        "co2":         co2_step,
        "nox":         nox_step,
        "pmx":         pmx_step,
        "fuel":        fuel_step,
        "collisions":  n_collisions,
        "teleports":   n_teleports,
    }, current_phase, phase_changes
 
 
def save_timeseries(seed_to_series, output_path):
    """
    Salva le serie temporali per ogni seed in un file .npz,
    una colonna per ogni metrica, con shape (n_seed, MAX_STEPS).
    """
    keys = list(next(iter(seed_to_series.values())).keys())
    stacked = {}
    for k in keys:
        arrs = []
        for s in seed_to_series.values():
            arr = np.array(s[k], dtype=float)
            if len(arr) < MAX_STEPS:
                arr = np.pad(arr, (0, MAX_STEPS - len(arr)), mode='edge')
            arrs.append(arr[:MAX_STEPS])
        stacked[k] = np.stack(arrs)  # shape (n_seed, MAX_STEPS)
    np.savez(output_path, **stacked)
    print(f"✓ Serie temporali salvate in: {output_path}")
 
 
def aggregate_episode(seed, series, phase_changes, total_reward=None,
                       action_counts=None):
    """Aggrega le serie temporali di un episodio in un dizionario riepilogativo."""
    row = {
        "seed":             seed,
        "avg_waiting_time": np.mean(series["waiting"]),
        "avg_queue_length":  np.mean(series["queue_len"]),
        "max_queue_length":  np.max(series["queue_max"]),
        "avg_speed":         np.mean(series["speed"]),
        "total_arrived":     np.sum(series["arrived"]),
        "phase_changes":     phase_changes,
        "avg_ped_waiting":   np.mean(series["ped_waiting"]),
        "total_co2":         np.sum(series["co2"]),
        "total_nox":         np.sum(series["nox"]),
        "total_pmx":         np.sum(series["pmx"]),
        "total_fuel":        np.sum(series["fuel"]),
        "total_collisions":  np.sum(series["collisions"]),
        "total_teleports":   np.sum(series["teleports"]),
    }
    if total_reward is not None:
        row["total_reward"] = total_reward
    if action_counts is not None:
        total = sum(action_counts.values())
        row["action_0_pct"] = action_counts[0] / total
        row["action_1_pct"] = action_counts[1] / total
    return row
 
 
def new_series():
    return {k: [] for k in
            ["waiting", "queue_len", "queue_max", "speed", "arrived",
             "ped_waiting", "co2", "nox", "pmx", "fuel",
             "collisions", "teleports"]}
 
 
def append_step(series, step_metrics):
    for k, v in step_metrics.items():
        series[k].append(v)
 
 
# ────────────────────────────────────────────────────────────────
# Modalità BASELINE — libera evoluzione, fasi fisse del semaforo
# ────────────────────────────────────────────────────────────────
def run_baseline(output_csv, output_timeseries):
    all_rows = []
    seed_to_series = {}
 
    for seed in SEEDS:
        print(f"\n[BASELINE] seed={seed}...")
 
        subprocess.run([
            "python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
            "-n", NET_FILE,
            "-e", str(MAX_STEPS),
            "-p", "3.0",
            "--persontrips",
            "-r", "pedoni.rou.xml",
            "--seed", str(seed),
        ], check=True, stdout=subprocess.DEVNULL)
 
        sumo_cmd = [
            "sumo", "-c", SUMO_CFG,
            "--seed", str(seed),
            "--waiting-time-memory", "1000",
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "100",
            "--collision.action", "teleport",
            "--collision.mingap-factor", "0",
            "--collision.check-junctions", "true",
        ]
 
        if traci.isLoaded():
            traci.close()
        traci.start(sumo_cmd)
 
        series        = new_series()
        prev_phase    = None
        phase_changes = 0
 
        for step in range(MAX_STEPS):
            traci.simulationStep()
            metrics, prev_phase, phase_changes = collect_step_metrics(prev_phase, phase_changes)
            append_step(series, metrics)
 
        traci.close()
 
        seed_to_series[seed] = series
        row = aggregate_episode(seed, series, phase_changes)
        all_rows.append(row)
        print(f"  arrived={row['total_arrived']:.0f} | "
              f"collisions={row['total_collisions']:.0f} | "
              f"phase_changes={row['phase_changes']}")
 
    # Salva serie temporali
    save_timeseries(seed_to_series, output_timeseries)
 
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ CSV baseline salvato in: {output_csv}")
    print(df.describe().round(2))
 
 
# ────────────────────────────────────────────────────────────────
# Modalità PPO — modello addestrato
# ────────────────────────────────────────────────────────────────
def run_ppo(model_path, stats_path, output_csv, output_timeseries):
    PPO, Monitor, DummyVecEnv, VecNormalize = _lazy_import_ppo()
 
    model = PPO.load(model_path)
    env   = DummyVecEnv([lambda: Monitor(MyEnv())])
    env   = VecNormalize.load(stats_path, env)
    env.training    = False
    env.norm_reward = True
 
    all_rows = []
    seed_to_series = {}
 
    for seed in SEEDS:
        print(f"\n[PPO] seed={seed}...")
 
        env.env_method("reset", seed=seed)
        obs = env.reset()
 
        series        = new_series()
        prev_phase    = None
        phase_changes = 0
        action_counts = {0: 0, 1: 0}
        total_reward  = 0.0
 
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action[0])] += 1
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
 
            metrics, prev_phase, phase_changes = collect_step_metrics(prev_phase, phase_changes)
            append_step(series, metrics)
 
            if done[0]:
                break
 
        seed_to_series[seed] = series
        row = aggregate_episode(seed, series, phase_changes,
                                 total_reward=total_reward,
                                 action_counts=action_counts)
        all_rows.append(row)
        print(f"  reward={total_reward:.1f} | arrived={row['total_arrived']:.0f} | "
              f"collisions={row['total_collisions']:.0f} | "
              f"phase_changes={row['phase_changes']}")
 
    env.close()
 
    # Salva serie temporali
    save_timeseries(seed_to_series, output_timeseries)
 
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ CSV PPO salvato in: {output_csv}")
    print(df.describe().round(2))
 
 
# ────────────────────────────────────────────────────────────────
# Entry point selezione
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cosa vuoi generare?")
    print("  1) CSV baseline (fasi fisse, libera evoluzione)")
    print("  2) CSV modello PPO")
    print("  3) CSV modello PPO con collisioni")
    choice = input("Scelta [1/2/3]: ").strip()
 
    if choice == "1":
        out_dir = "./csv_generati/baseline_opt"
        os.makedirs(out_dir, exist_ok=True)
        run_baseline(
            output_csv=os.path.join(out_dir, "metrics_baseline.csv"),
            output_timeseries=os.path.join(out_dir, "timeseries_baseline.npz")
        )
 
    elif choice == "2":
        from sumo_env import MyEnv
        model_path = "./models/t20energy-pedwaiting_f2/ppo_semaforo"
        stats_path = "./models/t20energy-pedwaiting_f2/vec_normalize_stats.pkl"
        out_dir    = "./csv_generati/PPO"
        os.makedirs(out_dir, exist_ok=True)
        run_ppo(
            model_path=model_path,
            stats_path=stats_path,
            output_csv=os.path.join(out_dir, "metrics_ppo.csv"),
            output_timeseries=os.path.join(out_dir, "timeseries_ppo.npz")
        )
    elif choice == "3":
        from sumo_env_collisions import MyEnv
        model_path = "./models/t20energy-pedwaiting-collisions_f2/ppo_semaforo"
        stats_path = "./models/t20energy-pedwaiting-collisions_f2/vec_normalize_stats.pkl"
        out_dir    = "./csv_generati/PPO-collisions"
        os.makedirs(out_dir, exist_ok=True)
        run_ppo(
            model_path=model_path,
            stats_path=stats_path,
            output_csv=os.path.join(out_dir, "metrics_ppo.csv"),
            output_timeseries=os.path.join(out_dir, "timeseries_ppo.npz")
        )
 
    else:
        print("Scelta non valida.")