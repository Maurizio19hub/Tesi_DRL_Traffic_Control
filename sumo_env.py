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
		self.observation_space = spaces.Box(low = 0, high = 1.0, shape=(19,), dtype = np.float32)

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

		# per la reward differenziale
		self.last_cost = 0


		self.in_yellow = False
		self.yellow_steps_remaining = 0
	
	
	#Resetta ambiente per nuova iterazione (episodio)
	##Cambiare il seed, o impostarlo passandolo alla funzione
	def reset(self, seed = None, options = None):
		super().reset(seed=seed)
		self.in_yellow = False
		self.yellow_steps_remaining = 0
		self.current_step = 0
		self.time_since_last_change = 0
		self.current_phase_index = 0
		self.episode += 1


		sumo_seed = seed if seed is not None else self.episode
		
		# Generazione percorsi Pedoni con nuovo seed
		subprocess.run([
			"python", os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py'),
			"-n", "incrocio3.net.xml",
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
		# forzare mantenimento della fase fino alla prossima decisione dell'agente
		traci.trafficlight.setPhase(self.tls_id, self.green_phases[self.current_phase_index])
		traci.trafficlight.setPhaseDuration(self.tls_id, self.max_steps)
		
		return self._get_observation(), {}

	def _get_observation(self):
		obs = []
		for branch_name, edge_ids in self.branches.items():
			
			total_halting = 0
			total_length = 0
			edge, num_lanes = edge_ids

			# ---CODE---
			total_halting = traci.edge.getLastStepHaltingNumber(edge)
			total_length = traci.lane.getLength(edge + "_0")


			# ---VELOCITÀ MEDIA---
			# serve per il calcolo del tempo medio di attraversamento
			veh_count = max(traci.edge.getLastStepVehicleNumber(edge), 1)
			mean_speed = traci.edge.getLastStepMeanSpeed(edge)
			# ---TEMPO DI ATTESA---

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
		'''
		if self.current_step % 500 == 0:
			print(f"[Step {self.current_step}] Pedoni in giro: {len(traci.person.getIDList())}, in attesa: {ped_waiting}")
		'''
		ped_norm = min(ped_waiting / self.max_pedestrians, 1.0)
		#obs.append(ped_norm)

		# ---TEMPO DALL'ULTIMO CAMBIO FASE---
		time_norm = min(self.time_since_last_change / self.max_green_duration, 1)
		obs.append(time_norm)

		# ---FASE CORRENTE---
		# usare one hot encoding
		phase_one_hot = [0, 0]
		phase_one_hot[self.current_phase_index] = 1
		obs.extend(phase_one_hot)
		'''
		# ---FASCIA ORARIA ---
		fascia_onehot = [0.0, 0.0, 0.0]
		if self.current_step < 3600:
			fascia_onehot[0] = 1.0   # mattina
		elif self.current_step < 7200:
			fascia_onehot[1] = 1.0   # pomeriggio
		else:
			fascia_onehot[2] = 1.0   # sera

		obs.extend(fascia_onehot)'''

		return np.array(obs, dtype=np.float32)
	
	def step(self, action):
		
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

		if self.current_step >= self.max_steps:
			traci.close()
			return np.zeros(19, dtype=np.float32), 0.0, True, False, {}

		obs    = self._get_observation()
		# aggiunta di un premio sul throughput
		reward = self._get_cost()

		terminated = self.current_step >= self.max_steps
		truncated  = False

		info = {
			"step":              self.current_step,
			"phase":             self.current_phase_index,
			"time_since_change": self.time_since_last_change,
			"sumo_phase":        traci.trafficlight.getPhase(self.tls_id),
		}

		if terminated:
			traci.close()
		'''
		print(f"step={self.current_step} action={action} "
		f"sumo_phase={traci.trafficlight.getPhase(self.tls_id)} "
		f"env_phase={self.current_phase_index} "
		f"time_since_change={self.time_since_last_change}")'''
		return obs, reward, terminated, truncated, info


	def close(self):
		if traci.isLoaded():
			traci.close()
	
	def _get_cost(self):
		queue_values   = []
		waiting_values = []

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
		
		current_cost = total_queue + 2*total_waiting
		ret = self.last_cost - current_cost
		if self.last_cost == 0: 
			self.last_cost = current_cost
			return 0
		self.last_cost = current_cost
		return (ret + balance_bonus*0.01)*20