import os
import traci
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess

class MyEnv(gym.Env):
	def __init__(self):
		super(self).__init__()
		
		#Fasi semaforiche
		self.action_space = spaces.Discrete(2) # keep / change
		
		#Dati presi dall'Enviroment (normalizzati):
		##Lunghezza code veicoli
		##Numero di pedoni
		##shape=(8,) indica solo 8 dati per le 4 code, 8 per gli attraversamenti pedonali, 8 per le walking areas	
		self.observation_space = spaces.Box(low = 0, high = 1.0, shape=(24,), dtype = np.float32)

		#File di configurazione SUMO
		self.sumo_cfg = "simulazione.sumocfg"

		#Contatore per la gestione dei seed
		self.episode = 0

		#Dizionario delle lane
		self.branches = {
			"nord" : [["nord_in_1", "nord_in2_1", "nord_in3_1"],["nord_in_2", "nord_in2_2", "nord_in3_2"]],
			"sud" : [["sud_in_1", "sud_in2_1", "sud_in3_1", "sud_in4_1"],["sud_in_2", "sud_in2_2", "sud_in3_2", "sud_in4_2"]],
			"est" : [["est_in_1"],["est_in_2"]],
			"ovest" : [["ovest_in_1"],["ovest_in_2"]]
		}
		self.branch_lengths = {}

		self.max_speed = 13.89 # massima velocità sulle strade, corrisponde a limite di 5O

		self.max_waiting_time = 300.0
		self.max_pedestrians = 20
	
	
	#Resetta ambiente per nuova iterazione (episodio)
	##Cambiare il seed, o impostarlo passandolo alla funzione
	def reset(self, seed = None, options = None):
		self.episode += 1
		sumo_seed = seed if seed is not None else self.episode

		#Generazione percorsi veicoli con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio.net.xml",
			"-e", "3600",
			"-p", "1.5",
			"-r", "veicoli.rou.xml", # Output file per veicoli
			"--seed", str(sumo_seed)
		], check=True, stdout=subprocess.DEVNULL)

		# Generazione percorsi Pedoni con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio.net.xml",
			"-e", "3600",
			"-p", "3.0",
			"--persontrips",
			"-r", "pedoni.rou.xml", # Output file per pedoni
			"--seed", str(sumo_seed)
		], check=True, stdout=subprocess.DEVNULL)

		
		sumo_cmd = [
			"sumo", "-c", self.sumo_cfg, # sumo-gui se vogli la modalità grafica
			"--seed", str(sumo_seed),
			"--waiting-time-memory", "1000", # serve per manternere memoria del tempo di attesa del veicolo per 1000s
			"--no-step-log", "true" # non riempie terminale
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
			for i in range(len(edge_ids[0])):

				# ---CODE---
				total_halting += traci.edge.getLastStepHaltingNumber(edge_ids[0][i])
				total_halting += traci.edge.getLastStepHaltingNumber(edge_ids[1][i])
				total_length += traci.lane.getLength(edge_ids[0][i])


				# ---VELOCITÀ MEDIA---
				# serve per il calcolo del tempo medio di attraversamento
				vehicles_lane_1 = traci.edge.getLastStepVehicleNumber(edge_ids[0][i])
				vehicles_lane_2 = traci.edge.getLastStepVehicleNumber(edge_ids[1][i])
				speed_lane_1 = traci.edge.getLastStepMeanSpeed(edge_ids[0][i])
				speed_lane_2 = traci.edge.getLastStepMeanSpeed(edge_ids[1][i])
				weighted_speed += vehicles_lane_1 * speed_lane_1 + vehicles_lane_1 * speed_lane_2
				total_vehicles += vehicles_lane_1 + vehicles_lane_2

				# ---TEMPO DI ATTESA---
				total_waiting += traci.edge.getWaitingTime(edge_ids[0][i]) + traci.edge.getWaitingTime(edge_ids[1][i])

			
			queue_meters = total_halting * 5 
			queue_norm = min(queue_meters / total_length, 1.0)

			if total_vehicles > 0:
				mean_speed = weighted_speed / total_vehicles
			else:
				mean_speed = self.max_speed

			speed_norm = mean_speed / self.max_speed

			waiting_norm = min(total_waiting / self.max_waiting_time, 1.0)

			obs.extend([queue_norm, speed_norm, waiting_norm])

		# ---TOTALE PEDONI IN ATTESA---
		ped_waiting = 0
		for person_id in traci.person.getIDList():
			if traci.person.getWaitingTime(person_id) > 0:
				ped_waiting += 1
		
		ped_norm = min(ped_waiting / self.max_pedestrians, 1.0)
		obs.append(ped_norm)