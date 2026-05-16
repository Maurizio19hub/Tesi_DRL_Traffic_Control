# baseline.py
# serve a verificare la reward media del semaforo a fasi fisse impostato sul file .net.xml, per confrontarla con quella di PPO
import traci
import numpy as np

SUMO_CFG = "simulazione.sumocfg"

branches = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}

max_speed        = 13.89
max_waiting_time = 90.0

def get_queue():
    total_queue = 0
    total_waiting = 0
    for branch_name, (edge, num_lanes) in branches.items():
        halting     = traci.edge.getLastStepHaltingNumber(edge)
        length      = traci.lane.getLength(edge + "_0")
        queue_norm  = min((halting * 5) / (length * num_lanes), 1.0)
        waiting_norm = min(traci.edge.getWaitingTime(edge) / max_waiting_time, 1.0)
        total_queue         += queue_norm
        total_waiting       += waiting_norm
    return total_queue, total_waiting

sumo_cmd = [
    "sumo", "-c", SUMO_CFG,
    "--no-step-log", "true",
    "--no-warnings", "true",
    "--waiting-time-memory", "1000",
]

traci.start(sumo_cmd)

prev_queue   = 0.0
prev_waiting = 0.0
total_reward = 0.0
step         = 0
decisions    = 0

while step < 10800:
    traci.simulationStep()
    step += 1

    if step % 5 == 0:  # stesso decision interval
        current_queue, current_waiting = get_queue()
        reward        = (prev_queue - current_queue) + 0.5 * (prev_waiting - current_waiting)
        total_reward += reward
        prev_queue    = current_queue
        prev_waiting  = current_waiting
        decisions += 1

traci.close()

print(f"Baseline semaforo fisso:")
print(f"  Total reward:              {total_reward:.2f}")
print(f"  Decisioni:                 {decisions}")
print(f"  Reward media per decisione:{total_reward / decisions:.5f}")