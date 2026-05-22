# baseline.py
import traci

SUMO_CFG = "simulazione.sumocfg"

branches = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}

max_waiting_time = 90.0


def get_cost():
    queue_values   = []
    waiting_values = []

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

    total_queue   = sum(queue_values)
    total_waiting = sum(waiting_values)

    balance_bonus = 0.0
    '''
    if all(q < QUEUE_THRESHOLD for q in queue_values):
        balance_bonus += 0.5
    if all(w < WAITING_THRESHOLD for w in waiting_values):
        balance_bonus += 0.2'''
    
    for i in range(len(queue_values)):
        balance_bonus += -(queue_values[i]-0.2)*0.1
        balance_bonus += -(waiting_values[i]-0.1)*0.1

    #print(f"TOTAL QUEUE : {total_queue}")
    #print(f"TOTAL WAITING : {total_waiting}")

    return ((total_queue + 2*total_waiting) - balance_bonus)

sumo_cmd = [
    "sumo", "-c", SUMO_CFG,
    "--no-step-log",  "true",
    "--no-warnings",  "true",
    "--waiting-time-memory", "1000",
]

traci.start(sumo_cmd)

total_reward = 0.0
step         = 0

while step < 3600:
    traci.simulationStep()
    step += 1
    total_reward += -get_cost()

traci.close()

print(f"Baseline semaforo fisso:")
print(f"  Total reward:           {total_reward:.2f}")
print(f"  Step:                   {step}")
print(f"  Reward media per step:  {total_reward / step:.5f}")