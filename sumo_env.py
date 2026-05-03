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
		self.observation_space = spaces.Box(low = 0, high = 1.0, shape=(15,), dtype = np.float32)

		#File di configurazione SUMO
		self.sumo_cfg = "simulazione.sumocfg"

		#Contatore per la gestione dei seed
		self.episode = 0

		#Dizionario delle lane
		self.branches = {
			"nord" : ["nord_in", "nord_in2", "nord_in3"],
			"sud" : ["sud_in", "sud_in2", "sud_in3", "sud_in4"],
			"est" : ["est_in"],
			"ovest" : ["ovest_in"]
		}
		#self.branch_lengths = {}

		self.max_speed = 13.89 # massima velocità sulle strade, corrisponde a limite di 5O

		self.max_waiting_time = 300.0
		self.max_pedestrians = 20

		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.green_phases = [0, 3]
		self.min_green_duration = 10
		self.yellow_duration = 6
		self.clearance_duration = 5
		self.tls_id = "tls_1"
		self.max_steps = 1000
		self.current_step = 0
	
	
	#Resetta ambiente per nuova iterazione (episodio)
	##Cambiare il seed, o impostarlo passandolo alla funzione
	def reset(self, seed = None, options = None):
		super().reset(seed=seed)
		self.current_step = 0
		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.episode += 1
		sumo_seed = seed if seed is not None else self.episode
		
		'''
		#Generazione percorsi veicoli con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio6.net.xml",
			"-e", "3600",
			"-p", "1.5",
			"-r", "veicoli.rou.xml", # Output file per veicoli
			"--seed", str(sumo_seed)
		], check=True, stdout=subprocess.DEVNULL)
		'''
		# Generazione percorsi Pedoni con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio6.net.xml",
			"-e", "3600",
			"-p", "3.0",
			"--persontrips",
			"-r", "pedoni.rou.xml", # Output file per pedoni
			"--seed", str(sumo_seed)
		], check=True, stdout=subprocess.DEVNULL)

		
		sumo_cmd = [
			"sumo-gui", "-c", self.sumo_cfg, # sumo-gui se vogli la modalità grafica
			"--seed", str(sumo_seed),
			"--waiting-time-memory", "1000", # serve per manternere memoria del tempo di attesa del veicolo per 1000s
			"--no-step-log", "true", # non riempie terminale
			"--start", "true",  # avvia automaticamente senza premere play (sumo-gui)
			"--delay", "500"  # 100ms tra ogni step = velocità normal
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
			weighted_speed = 0.0
			total_waiting = 0.0
			for edge in edge_ids:

				# ---CODE---
				total_halting += traci.edge.getLastStepHaltingNumber(edge)
				total_length += traci.lane.getLength(edge + "_1")


				# ---VELOCITÀ MEDIA---
				# serve per il calcolo del tempo medio di attraversamento
				vehicles = traci.edge.getLastStepVehicleNumber(edge)
				speed = traci.edge.getLastStepMeanSpeed(edge)
				weighted_speed += vehicles * speed
				total_vehicles += vehicles
				# ---TEMPO DI ATTESA---
				total_waiting += traci.edge.getWaitingTime(edge)

			
			queue_meters = total_halting * 5 
			queue_norm = min(queue_meters / total_length*2, 1.0)

			if total_vehicles > 0:
				mean_speed = weighted_speed / total_vehicles
			else:
				mean_speed = self.max_speed

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

		return np.array(obs, dtype=np.float32)
	
	def step(self, action):
		if action == 1 and self.time_since_last_change > self.min_green_duration:

			clearance_phase = self.green_phases[self.current_phase_index] + 1
			traci.trafficlight.setPhase(self.tls_id, clearance_phase)
			for _ in range(self.clearance_duration):
				traci.simulationStep()
				self.current_step += 1
				self.time_since_last_change += 1

			yellow_phase = self.green_phases[self.current_phase_index] + 2
			traci.trafficlight.setPhase(self.tls_id, yellow_phase)
			for _ in range(self.yellow_duration):
				traci.simulationStep()
				self.current_step += 1
				self.time_since_last_change += 1
			
			self.current_phase_index = 1 - self.current_phase_index
			traci.trafficlight.setPhase(
				self.tls_id,
				self.green_phases[self.current_phase_index]
			)
			self.time_since_last_change = 0
		else:
			traci.simulationStep()
			self.current_step += 1
			self.time_since_last_change += 1

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
		pass

	def close(self):
		if traci.isLoaded():
			traci.close()