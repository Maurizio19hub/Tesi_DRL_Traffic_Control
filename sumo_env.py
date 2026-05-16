import os
import traci
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess

class MyEnv(gym.Env):
	def __init__(self):
		super().__init__()
		
		#Fasi semaforiche
		self.action_space = spaces.Discrete(2) # keep / change
		
		#Dati presi dall'Enviroment (normalizzati):
		##Lunghezza code veicoli
		##Numero di pedoni
		self.observation_space = spaces.Box(low = 0, high = 1.0, shape=(18,), dtype = np.float32)

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

		self.max_speed = 13.89 # massima velocità sulle strade, corrisponde a limite di 5O

		self.max_waiting_time = 300.0
		self.max_pedestrians = 20

		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.green_phases = [0, 2]
		self.min_green_duration = 15
		self.yellow_duration = 3
		self.tls_id = "tls_1"
		self.max_steps = 10800
		self.current_step = 0

		# per la reward differenziale sulla lunghezza delle code
		self.previous_queue = 0.0
		self.DECISION_INTERVAL =  5
	
	
	#Resetta ambiente per nuova iterazione (episodio)
	##Cambiare il seed, o impostarlo passandolo alla funzione
	def reset(self, seed = None, options = None):
		super().reset(seed=seed)
		self.current_step = 0
		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.episode += 1

		self.previous_queue = 0.0

		sumo_seed = seed if seed is not None else self.episode
		
		# Generazione percorsi Pedoni con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio3.net.xml",
			"-e", "10800",
			"-p", "3.0",
			"--persontrips",
			"-r", "pedoni.rou.xml", # Output file per pedoni
			"--seed", str(sumo_seed)
		], check=True, stdout=subprocess.DEVNULL)

		
		sumo_cmd = [
			"sumo", "-c", self.sumo_cfg, # sumo-gui se vogli la modalità grafica
			"--seed", str(sumo_seed),
			"--waiting-time-memory", "1000", # serve per manternere memoria del tempo di attesa del veicolo per 1000s
			"--no-step-log", "true", # non riempie terminale
			#"--start", "true",  # avvia automaticamente senza premere play (sumo-gui)
			#"--delay", "100",  # 100ms tra ogni step = velocità normal
			"--time-to-teleport", "100",
			"--collision.action", "teleport",
			"--collision.mingap-factor", "0",
			"--collision.check-junctions", "true",
			"--no-warnings", "true", # warnings in fase di training intasano il terminale
		]

		if traci.isLoaded(): traci.close() # chiude le istanze già avviate se esistono 
		traci.start(sumo_cmd)
		
		return self._get_observation(), {}

	def _get_observation(self):
		obs = []
		for branch_name, edge_ids in self.branches.items():
			
			total_halting = 0
			total_length = 0
			total_vehicles = 0
			total_waiting = 0.0
			edge, num_lanes = edge_ids

			# ---CODE---
			total_halting = traci.edge.getLastStepHaltingNumber(edge)
			total_length = traci.lane.getLength(edge + "_0")


			# ---VELOCITÀ MEDIA---
			# serve per il calcolo del tempo medio di attraversamento
			total_vehicles = traci.edge.getLastStepVehicleNumber(edge)
			mean_speed = traci.edge.getLastStepMeanSpeed(edge)
			# ---TEMPO DI ATTESA---
			total_waiting = traci.edge.getWaitingTime(edge)

			
			queue_meters = total_halting * 5 # lunghezza veicolo 
			queue_norm = min(queue_meters / (total_length*num_lanes), 1.0)

			speed_norm = min(mean_speed / self.max_speed, 1.0)

			waiting_norm = min(total_waiting / self.max_waiting_time, 1.0)

			obs.extend([queue_norm, speed_norm, waiting_norm])

		# ---TOTALE PEDONI IN ATTESA---
		ped_waiting = 0
		for person_id in traci.person.getIDList():
			if traci.person.getWaitingTime(person_id) > 0:
				ped_waiting += 1
		
		ped_norm = min(ped_waiting / self.max_pedestrians, 1.0)
		obs.append(ped_norm)

		# ---TEMPO DALL'ULTIMO CAMBIO FASE---
		time_norm = min(self.time_since_last_change / self.min_green_duration, 1)
		obs.append(time_norm)

		# ---FASE CORRENTE---
		obs.append(self.current_phase_index)

		# ---FASCIA ORARIA ---
		fascia_onehot = [0.0, 0.0, 0.0]
		if self.current_step < 3600:
			fascia_onehot[0] = 1.0   # mattina
		elif self.current_step < 7200:
			fascia_onehot[1] = 1.0   # pomeriggio
		else:
			fascia_onehot[2] = 1.0   # sera

		obs.extend(fascia_onehot)

		return np.array(obs, dtype=np.float32)
	
	def step(self, action):
		if action == 1 and self.time_since_last_change > self.min_green_duration:

			yellow_phase = self.green_phases[self.current_phase_index] + 1
			traci.trafficlight.setPhase(self.tls_id, yellow_phase)
			for _ in range(self.yellow_duration):
				traci.simulationStep()
				self.current_step += 1
				self.time_since_last_change += 1
				if self.current_step >= self.max_steps:
					traci.close()
					# restituisci subito senza calcolare obs/reward
					return np.zeros(15, dtype=np.float32), 0.0, True, False, {}
			
			self.current_phase_index = 1 - self.current_phase_index
			traci.trafficlight.setPhase(
				self.tls_id,
				self.green_phases[self.current_phase_index]
			)
			self.time_since_last_change = 0

			remaining = self.DECISION_INTERVAL - self.yellow_duration
			for _ in range(remaining):
				traci.simulationStep()
				self.current_step += 1
				if self.current_step >= self.max_steps:
					traci.close()
					return np.zeros(18, dtype=np.float32), 0.0, True, False, {}
		else:
			for _ in range(self.DECISION_INTERVAL):
				traci.simulationStep()
				self.current_step += 1
				self.time_since_last_change += 1
				if self.current_step >= self.max_steps:
					traci.close()
					return np.zeros(18, dtype=np.float32), 0.0, True, False, {}


		# ---CALCOLO OSSERVAZIONE E REWARD---
		obs = self._get_observation()
		reward = self._get_reward()

		# ---CONTROLLO SE EPISODIO È TERMINATO---
		terminated = self.current_step >= self.max_steps
		truncated = False
		
		info = {
			"step": self.current_step,
			"phase": self.current_phase_index,
			"time_since_change": self.time_since_last_change
		}

		if terminated: traci.close()

		return obs, reward, terminated, truncated, info


	def _get_reward(self):
		current_queue = 0
		for branch_name, edge_ids in self.branches.items():
			edge, num_lanes = edge_ids
			total_halting = traci.edge.getLastStepHaltingNumber(edge)
			total_length = traci.lane.getLength(edge + "_0")
			queue_norm = min((total_halting * 5) / (total_length * num_lanes), 1.0)
			current_queue += queue_norm
		reward = self.previous_queue - current_queue
		self.previous_queue = current_queue
		return reward 


	def close(self):
		if traci.isLoaded():
			traci.close()