import os
import traci
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess

OBS_SHAPE = 20

class MyEnv(gym.Env):
	def __init__(self):
		super().__init__()
	
		self.action_space = spaces.Discrete(2) # keep / change
		self.observation_space = spaces.Box(low = 0, high = 1.0, shape=(OBS_SHAPE,), dtype = np.float32)

		#File di configurazione SUMO
		self.sumo_cfg = "simulazione.sumocfg"

		#Contatore per la gestione dei seed
		self.episode = 0

		#Dizionario delle lane
		self.branches = {
			"nord":  ("nord_in",  1),
			"sud":   ("sud_in",   1),
			"est":   ("est_in",   2),
			"ovest": ("ovest_in", 2),
		}


		self.outbound_map = {
			"nord_in": "nord_out",
			"sud_in": "sud_out",
			"est_in":  "est_out",
			"ovest_in":  "ovest_out"
		}

		self.max_speed = 13.89 # massima velocità sulle strade, corrisponde a limite di 5O

		self.max_waiting_time = 90.0
		self.max_pedestrians = 5

		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.green_phases = [0, 2]
		self.min_green_duration = 15
		self.max_green_duration = 120
		self.yellow_duration = 3
		self.tls_id = "tls_1"
		self.max_steps = 3600
		self.current_step = 0

		self.last_cost = 0


		self.in_yellow = False
		self.yellow_steps_remaining = 0

		# metriche per la valutazione da salvare in CSV
		self.prev_emissions = {}  # chiave: veh_id, valore: dict con co2, nox, pmx, fuel

		self.episode_waiting_times = []
		self.episode_queue_lengths = []
		self.episode_queue_max     = []
		self.episode_speeds        = []
		self.episode_arrived       = 0
		self.episode_ped_waiting   = []
		self.co2_per_step      = []   
		self.nox_per_step      = []   
		self.pmx_per_step      = []   
		self.fuel_per_step     = []   
		self.episode_collisions    = 0
		self.episode_teleports     = 0
		self.episode_phase_changes = 0
		self.prev_phase            = None
	
	
	#Resetta ambiente per nuova iterazione (episodio)
	def reset(self, seed = None, options = None):
		super().reset(seed=seed)

		self.prev_emissions.clear()

		self.episode_waiting_times = []
		self.episode_queue_lengths = []
		self.episode_queue_max     = []
		self.episode_speeds        = []
		self.episode_arrived       = 0
		self.episode_ped_waiting   = []
		self.co2_per_step      = []   
		self.nox_per_step      = []   
		self.pmx_per_step      = []   
		self.fuel_per_step     = []   
		self.episode_collisions    = 0
		self.episode_teleports     = 0
		self.episode_phase_changes = 0
		self.prev_phase            = None

		self.in_yellow = False
		self.yellow_steps_remaining = 0
		self.current_step = 0
		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.episode += 1
		self.last_cost = 0


		sumo_seed = seed if seed is not None else self.episode
		
		# Generazione percorsi Pedoni con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio3_300mf.net.xml",
			"-e", "3600",
			"-p", "3.0",
			"--persontrips",
			"-r", "pedoni.rou.xml", # Output file per pedoni
			"--seed", str(sumo_seed)
		], check=True, stdout=subprocess.DEVNULL)

		
		sumo_cmd = [
			"sumo", "-c", self.sumo_cfg,
			"--seed", str(sumo_seed),
			"--waiting-time-memory", "1000", 
			"--no-step-log", "true",
			"--time-to-teleport", "100",
			"--collision.action", "teleport",
			"--collision.mingap-factor", "0",
			"--collision.check-junctions", "true",
			"--no-warnings", "true",
		]

		if traci.isLoaded(): traci.close() 
		traci.start(sumo_cmd)
		traci.trafficlight.setPhase(self.tls_id, self.green_phases[self.current_phase_index])
		traci.trafficlight.setPhaseDuration(self.tls_id, self.max_steps)
		
		return self._get_observation(), {}

	def _get_observation(self):
		obs = []
		for branch_name, edge_ids in self.branches.items():
			
			total_halting = 0
			total_length = 0
			edge, num_lanes = edge_ids

			total_halting = traci.edge.getLastStepHaltingNumber(edge)
			total_length = traci.lane.getLength(edge + "_0")

			veh_count = max(traci.edge.getLastStepVehicleNumber(edge), 1)
			mean_speed = traci.edge.getLastStepMeanSpeed(edge)
	
			incoming_norm = min(veh_count / (num_lanes * 10), 1.0)

			queue_meters = total_halting * 5 # lunghezza veicolo 
			queue_norm = min(queue_meters / (total_length*num_lanes), 1.0)

			speed_norm = min(mean_speed / self.max_speed, 1.0)

			waiting_mean = traci.edge.getWaitingTime(edge) / veh_count
			waiting_norm = min(waiting_mean / self.max_waiting_time, 1.0)

			obs.extend([queue_norm, speed_norm, waiting_norm, incoming_norm])

		# ---TOTALE PEDONI IN ATTESA---
		ped_waiting = 0
		for person_id in traci.person.getIDList():
			if traci.person.getWaitingTime(person_id) > 0:
				ped_waiting += 1
		
		ped_norm = min(ped_waiting / self.max_pedestrians, 1.0)
		obs.append(ped_norm)

		# ---TEMPO DALL'ULTIMO CAMBIO FASE---
		time_norm = min(self.time_since_last_change / self.max_green_duration, 1)
		obs.append(time_norm)

		# ---FASE CORRENTE---
		phase_one_hot = [0, 0]
		phase_one_hot[self.current_phase_index] = 1
		obs.extend(phase_one_hot)

		return np.array(obs, dtype=np.float32)
	
	def step(self, action):
		co2_per_step_val = 0.0
		nox_per_step_val = 0.0
		pmx_per_step_val = 0.0
		fuel_per_step_val = 0.0
		
		# Applica l'azione solo se siamo in verde e il tempo minimo è rispettato
		if action == 1 and self.time_since_last_change >= self.min_green_duration and not self.in_yellow:
			# Inizia la transizione — imposta giallo
			self.in_yellow = True
			self.yellow_steps_remaining = self.yellow_duration
			yellow_phase = self.green_phases[self.current_phase_index] + 1
			traci.trafficlight.setPhase(self.tls_id, yellow_phase)
			traci.trafficlight.setPhaseDuration(self.tls_id, self.max_steps)
		else:
			# Se siamo in fase gialla, scalare il countdown
			if self.in_yellow:
				self.yellow_steps_remaining -= 1
				if self.yellow_steps_remaining == 0:
					# Fine giallo → passa al verde successivo
					self.in_yellow = False
					self.current_phase_index = 1 - self.current_phase_index
					traci.trafficlight.setPhase(self.tls_id, self.green_phases[self.current_phase_index])
					traci.trafficlight.setPhaseDuration(self.tls_id, self.max_steps)
					self.time_since_last_change = 0
			else:
				# Verde normale
				traci.trafficlight.setPhase(self.tls_id, self.green_phases[self.current_phase_index])
				traci.trafficlight.setPhaseDuration(self.tls_id, self.max_steps)
				self.time_since_last_change += 1
			
		# Avanza di un secondo
		traci.simulationStep()
		self.current_step += 1

		w_sum, q_sum, q_max_val, s_sum, s_count = 0, 0, 0, 0, 0
		co2_step, nox_step, pmx_step, fuel_step = 0.0, 0.0, 0.0, 0.0

		for branch_name, (edge, num_lanes) in self.branches.items():  
			veh_ids = traci.edge.getLastStepVehicleIDs(edge)
			n_veh = len(veh_ids)
			halting = traci.edge.getLastStepHaltingNumber(edge)
			
			w_sum += traci.edge.getWaitingTime(edge)
			
			q_m = halting * 5
			q_sum += q_m
			q_max_val = max(q_max_val, q_m)
			
			if n_veh > 0:
				s_sum += traci.edge.getLastStepMeanSpeed(edge) * n_veh
				s_count += n_veh
			
			for veh_id in veh_ids:
				co2_per_step_val += traci.vehicle.getCO2Emission(veh_id)
				nox_per_step_val += traci.vehicle.getNOxEmission(veh_id)
				pmx_per_step_val += traci.vehicle.getPMxEmission(veh_id)
				fuel_per_step_val += traci.vehicle.getFuelConsumption(veh_id)

		self.co2_per_step.append(co2_per_step_val)
		self.nox_per_step.append(nox_per_step_val)
		self.pmx_per_step.append(pmx_per_step_val)
		self.fuel_per_step.append(fuel_per_step_val)

		self.episode_waiting_times.append(w_sum / max(s_count, 1))
		self.episode_queue_lengths.append(q_sum / 4)  # assumi 4 corsie
		self.episode_queue_max.append(q_max_val)
		self.episode_speeds.append(s_sum / max(s_count, 1))
		self.episode_arrived += traci.simulation.getArrivedNumber()

		self.episode_collisions += len(traci.simulation.getCollisions())
		self.episode_teleports += traci.simulation.getStartingTeleportNumber()

		ped_w = sum(1 for pid in traci.person.getIDList() if traci.person.getWaitingTime(pid) > 0)
		self.episode_ped_waiting.append(ped_w)
		
		current_phase = traci.trafficlight.getPhase("tls_1")
		if self.prev_phase is not None and current_phase != self.prev_phase:
			self.episode_phase_changes += 1
		self.prev_phase = current_phase
		
		
		info = {
			"step":              self.current_step,
			"phase":             self.current_phase_index,
			"time_since_change": self.time_since_last_change,
			"sumo_phase":        traci.trafficlight.getPhase(self.tls_id),
		}
		
		if self.current_step >= self.max_steps:
			metrics = {
				'mean_queue':         np.mean(self.episode_queue_lengths) if self.episode_queue_lengths else 0,
				'max_queue':          max(self.episode_queue_max) if self.episode_queue_max else 0,
				'mean_waiting_time':  np.mean(self.episode_waiting_times) if self.episode_waiting_times else 0,
				'mean_speed':         np.mean(self.episode_speeds) if self.episode_speeds else 0,
				"total_co2":           np.sum(self.co2_per_step),
				"total_nox":           np.sum(self.nox_per_step),
				"total_pmx":           np.sum(self.pmx_per_step),
				"total_fuel":          np.sum(self.fuel_per_step),	
				'arrived_vehicles':   self.episode_arrived,
				'pedestrian_waiting': np.mean(self.episode_ped_waiting) if self.episode_ped_waiting else 0,
				'phase_changes':      self.episode_phase_changes,
				'collisions':         self.episode_collisions,
				'teleports':          self.episode_teleports,
			}
			info['physical_metrics'] = metrics
			traci.close()
			return np.zeros(OBS_SHAPE, dtype=np.float32), 0.0, True, False, info
		
		# calcolo observation e reward
		obs    = self._get_observation()
		reward = self._get_reward()

		terminated = self.current_step >= self.max_steps
		truncated  = False

		
		
		if terminated:
			metrics = {
				'mean_queue':         np.mean(self.episode_queue_lengths) if self.episode_queue_lengths else 0,
				'max_queue':          max(self.episode_queue_max) if self.episode_queue_max else 0,
				'mean_waiting_time':  np.mean(self.episode_waiting_times) if self.episode_waiting_times else 0,
				'mean_speed':         np.mean(self.episode_speeds) if self.episode_speeds else 0,
				"total_co2":           np.sum(self.co2_per_step),
				"total_nox":           np.sum(self.nox_per_step),
				"total_pmx":           np.sum(self.pmx_per_step),
				"total_fuel":          np.sum(self.fuel_per_step),		
				'arrived_vehicles':   self.episode_arrived,
				'pedestrian_waiting': np.mean(self.episode_ped_waiting) if self.episode_ped_waiting else 0,
				'phase_changes':      self.episode_phase_changes,
				'collisions':         self.episode_collisions,
				'teleports':          self.episode_teleports,
			}
			info['physical_metrics'] = metrics
			traci.close()
		
		return obs, reward, terminated, truncated, info
	

	def close(self):
		if traci.isLoaded():
			traci.close()
	
	def _get_reward(self):
		queue_values   = []
		waiting_values = []
		total_throughput = 0.0

		for branch_name, edge_ids in self.branches.items():
			edge, num_lanes = edge_ids
			total_halting = traci.edge.getLastStepHaltingNumber(edge)
			total_length  = traci.lane.getLength(edge + "_0")
			queue_norm    = min((total_halting * 5) / (total_length * num_lanes), 1.0)
			queue_values.append(queue_norm)

			veh_count    = max(traci.edge.getLastStepVehicleNumber(edge), 1)
			waiting_mean = traci.edge.getWaitingTime(edge) / veh_count
			waiting_norm = min(waiting_mean / self.max_waiting_time, 1.0)
			waiting_values.append(waiting_norm)

			total_throughput += traci.edge.getLastStepVehicleNumber(self.outbound_map[edge])

		total_queue   = sum(queue_values)
		total_waiting = sum(waiting_values)

		balance_penalty = 0.0
		
		for i in range(len(queue_values)):
			if queue_values[i] > 0.26:
				balance_penalty += -(queue_values[i]-0.26)*0.75
			if waiting_values[i] > 0.4:
				balance_penalty += -(waiting_values[i]-0.3)*0.75

		energy_penalty = self._get_energy_penalty()

		# ---TOTALE PEDONI IN ATTESA---
		ped_waiting = 0
		for person_id in traci.person.getIDList():
			if traci.person.getWaitingTime(person_id) > 0:
				ped_waiting += 1
		
		ped_norm = min(ped_waiting / self.max_pedestrians, 1.0)

		
		
		current_cost = (total_queue*2.5) + (2*total_waiting) + (energy_penalty*0.01) + (ped_norm * 0.2)
		ret = self.last_cost - current_cost

		if self.last_cost == 0: 
			self.last_cost = current_cost
			return 0
		self.last_cost = current_cost

		long_green_penalty = -max(0, self.time_since_last_change - 60) * 0.005

		return (ret + balance_penalty)*20 + long_green_penalty


	def _get_energy_penalty(self):
		energy_cost = 0.0
		for branch_name, edge_ids in self.branches.items():
			edge, num_lanes = edge_ids
			veh_count  = traci.edge.getLastStepVehicleNumber(edge)
			if veh_count == 0:
				continue
			mean_speed = traci.edge.getLastStepMeanSpeed(edge)
			halting    = traci.edge.getLastStepHaltingNumber(edge)
			# Veicoli fermi — idle consumption
			idle_cost   = halting * 0.5
			# Veicoli lenti ma in movimento
			moving      = veh_count - halting
			moving_cost = moving * (1 - mean_speed / self.max_speed) * 0.2
			energy_cost += idle_cost + moving_cost
			
		return energy_cost