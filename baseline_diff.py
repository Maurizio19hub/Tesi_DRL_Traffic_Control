import traci
import numpy as np

# --- CONFIGURAZIONE BASELINE ---
GREEN_DURATION = 34 
YELLOW_DURATION = 3 
SCALING_FACTOR = 20    # Il tuo moltiplicatore x20
BONUS_WEIGHT = 0.01    # Il peso 0.01 che hai impostato nell'ultimo step

max_waiting_time = 90.0
balance_bonus = 0

branches = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}

def run_fixed_baseline(sumo_cfg):
    # Avvia SUMO in modalità senza GUI per velocità (usa 'sumo-gui' per vedere)
    traci.start(["sumo", "-c", sumo_cfg])
    
    total_reward = 0
    last_cost = 0
    step = 0
    
    # Sequenza fasi dal tuo XML: 0(G), 1(Y), 2(G), 3(Y)
    phases = [1,3]
    current_phase_idx = 0
    phase_timer = 0

    print(f"Inizio baseline per {"tls_1"} (Verde: 34s, Giallo: 3s)...")

    while traci.simulation.getMinExpectedNumber() > 0 and step < 3600:
        traci.simulationStep()
        
        # --- LOGICA SEMAFORO FISSO ---
        phase_timer += 1
        current_phase = phases[current_phase_idx]
        
        # Controllo cambio fase in base al tipo (Verde vs Giallo)
        if current_phase in [3]: 
            if phase_timer >= 34: # durata verde dal tuo XML
                current_phase_idx = (current_phase_idx + 1) % len(phases)
                phase_timer = 0
        # Fasi 1 e 3 sono gialle nel tuo XML
        else: 
            if phase_timer >= 3: # durata giallo dal tuo XML
                current_phase_idx = (current_phase_idx + 1) % len(phases)
                phase_timer = 0
        
        traci.trafficlight.setPhase("tls_1", phases[current_phase_idx])

        # --- CALCOLO REWARD (TUA LOGICA AGGIORNATA) ---
        # 1. Recupero dati dai bracci (es. Nord, Sud, Est, Ovest)
        queue_values = [] # Inserisci qui le tue densità normalizzate per corsia
        waiting_values = [] # Inserisci qui i tuoi tempi di attesa normalizzati
        
        for branch_name, edge_ids in branches.items():
            edge, num_lanes = edge_ids
            total_halting = traci.edge.getLastStepHaltingNumber(edge)
            total_length  = traci.lane.getLength(edge + "_0")
            queue_norm    = min((total_halting * 5) / (total_length * num_lanes), 1.0)
            queue_values.append(queue_norm)

            veh_count    = max(traci.edge.getLastStepVehicleNumber(edge), 1)
            waiting_mean = traci.edge.getWaitingTime(edge) / veh_count
            waiting_norm = min(waiting_mean / max_waiting_time, 1.0)
            waiting_values.append(waiting_norm)
        
        total_queue = sum(queue_values)
        total_waiting = sum(waiting_values)
        
        # 2. Calcolo del bilanciamento (Penalty)
        balance_bonus = 0
        for i in range(len(queue_values)):
            balance_bonus += -(queue_values[i] - 0.2) * 0.1
            balance_bonus += -(waiting_values[i] - 0.1) * 0.1

        # 3. Delta Reward: Costo(t-1) - Costo(t)
        current_cost = total_queue + 2 * total_waiting
        
        if step == 0:
            reward = 0
        else:
            delta_improvement = last_cost - current_cost
            # La tua formula finale: (ret + bonus*0.01) * 20
            reward = (delta_improvement + balance_bonus * BONUS_WEIGHT) * SCALING_FACTOR
            total_reward += reward

        last_cost = current_cost
        step += 1

    traci.close()
    print(f"--- RISULTATO BASELINE ---")
    print(f"Reward Totale Episodio: {total_reward:.2f}")
    return total_reward

run_fixed_baseline("simulazione.sumocfg")