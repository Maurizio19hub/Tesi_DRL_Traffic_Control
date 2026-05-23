import traci

SUMO_CFG = "simulazione.sumocfg"

branches = {
    "nord":  ("nord_in",  1),
    "sud":   ("sud_in",   1),
    "est":   ("est_in",   2),
    "ovest": ("ovest_in", 2),
}

max_waiting_time = 90.0

def get_cost(last_cost):
    queue_values   = []
    waiting_values = []

    for branch_name, (edge, num_lanes) in branches.items():
        halting      = traci.edge.getLastStepHaltingNumber(edge)
        length       = traci.lane.getLength(edge + "_0")
        queue_norm   = min((halting * 5) / (length * num_lanes), 1.0)
        queue_values.append(queue_norm)

        veh_count    = max(traci.edge.getLastStepVehicleNumber(edge), 1)
        waiting_mean = traci.edge.getWaitingTime(edge) / veh_count
        waiting_norm = min(waiting_mean / max_waiting_time, 1.0)
        waiting_values.append(waiting_norm)

    total_queue   = sum(queue_values)
    total_waiting = sum(waiting_values)

    balance_bonus = 0.0
    for i in range(len(queue_values)):
        if queue_values[i] > 0.4:
            balance_bonus += -(queue_values[i] - 0.4) * 0.5
        if waiting_values[i] > 0.4:
            balance_bonus += -(waiting_values[i] - 0.3) * 0.5

    current_cost = total_queue + 2 * total_waiting

    if last_cost == 0:
        return 0, current_cost

    ret = last_cost - current_cost
    reward = (ret + balance_bonus) * 20
    return reward, current_cost


sumo_cmd = [
    "sumo", "-c", SUMO_CFG,
    "--seed", "42",
    "--no-step-log",  "true",
    "--no-warnings",  "true",
    "--waiting-time-memory", "1000",
    "--time-to-teleport", "100",
    "--collision.action", "teleport",
    "--collision.mingap-factor", "0",
    "--collision.check-junctions", "true",
]

traci.start(sumo_cmd)

total_reward = 0.0
last_cost    = 0.0
step         = 0

while step < 3600:
    traci.simulationStep()
    step += 1
    reward, last_cost = get_cost(last_cost)
    total_reward += reward

traci.close()

print(f"Baseline semaforo fisso:")
print(f"  Total reward:          {total_reward:.2f}")
print(f"  Step:                  {step}")
print(f"  Reward media per step: {total_reward / step:.5f}")