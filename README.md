# Versioni software usate per lo sviluppo

- **Python**: 3.14.3
- **SUMO**: Eclipse SUMO v1.26.0+0485-c820d16e9df (Build: Linux-6.18.9-200.fc43.x86_64 x86_64 GNU 15.2.1 Release)
- **Sistema operativo**: Linux (Fedora 43)

# Struttura della Cartella

.
├── baseline_diff.py              # Confronto tra baseline e modello DRL
├── baseline.py                   # Simulazione baseline (tempi fissi)
├── debug.py                      # Script di debug per l'ambiente
├── fine_tuning.py                # Fine-tuning di modelli pre-addestrati
├── get_fixed_stats.py            # Estrazione statistiche da simulazione fissa
├── get_model_stats.py            # Estrazione statistiche da modello DRL
├── incrocio3_300mf.net.xml       # Rete stradale SUMO (incrocio a 4 vie)
├── main.py                       # Script principale di training
├── models/                       # Modelli addestrati e checkpoint
│   ├── correct_time/             # Modelli con tempi corretti
│   ├── t1/ ... t20*/             # Serie di esperimenti di training
│   └── ppo_semaforo.zip          # Modello PPO finale
├── new_sumo_flow_gen/            # Generatore flussi di traffico calibrati
│   ├── flow_gen.py
│   ├── flussi_calibrati_*.rou.xml
│   └── turn_ratios.csv
├── pedoni.rou.xml                # Flussi pedonali
├── random_flow_gen/              # Generatore flussi casuali
│   └── random.py
├── results/                      # Risultati e visualizzazioni
│   ├── baseline/                 # Metriche baseline
│   ├── ppo_*/                    # Metriche modelli DRL
│   └── res.py                    # Script di analisi risultati
├── simulazione.sumocfg           # Configurazione principale SUMO
├── sumo_env.py                   # Ambiente Gymnasium custom (MyEnv)
├── test.py                       # Test dell'ambiente
├── tls.tll.xml                   # Programma semaforico di riferimento
├── trips.trips.xml               # Definizione trips veicolari
└── vecchie_configurazioni/      # Backup configurazioni precedenti
    ├── fileNET/                  # Reti stradali alternative
    ├── fileOSM/                  # Mappe OpenStreetMap
    ├── flussi_calibrati.rou.xml
    └── sumo_flow_generator/      # Tool generazione flussi legacy

# Dipendenze

## Python Standard Library
- csv
- collections (defaultdict)
- datetime
- os
- subprocess

## Scientific Computing
- numpy
- pandas
- matplotlib
- matplotlib.gridspec

## SUMO
- traci

## Reinforcement Learning
- gymnasium
- stable-baselines3
- stable-baselines3.common.monitor
- stable_baselines3.common.vec_env
- stable_baselines3.common.callbacks
- stable_baselines3.common.evaluation

## Environment
- sumo_env (modulo locale, contiene MyEnv)

# Comandi
- Avvio training : python main.py
- Comando testing : python debug.py
- Comando generazione metriche e grafici (PPO) : python get_model_stats.py
- Comando generazione metriche e grafici (baseline) : python get_fixed_stats.py
- Comando generazione grafico reward : cd results -> python res.py

# Funzione di Reward

## Variabili

Funzione python che calcola la reward : _get_cost(self).
Preleva tramite traci i dati dalla simulazione per calcolare:
- total_waitin : tempo di attesa normalizzato (somma di tutte le strade)
- total_queue : lunghezza delle code normalizzata (somma di tutte le strade)
- energy_penalty : calcolata tramite funzione _get_energy_penalty(self), che combinate penalità per veicoli fermi e una penalità sui veicoli in movimento che diminuisce all'aumentare della velocità dei veicoli. Questo favorisce il mantenimento del flusso veicolare a velocità sostenute, agendo indirettamente sulle accelerazioni e decelerazioni e quindi sul consumo energetico e le emissioni.
- ped_norm : pedoni in attesa (normalizzato)
- long_green_penalty (escluso da reward differenziale) : penalità incrementale quando l'agente mantiene la fase per più di 60 secondi
- balance_bonus (escluso da reward differenziale) : penalità incrementale che diventa un bonus se l'agente mantiene i valori delle code e dei tempi di attesa sotto una certa soglio 

## Calcolo della Reward

Reward viene calcolata come differenza tra costo precedente e costo corrente, assume valori positivi se il costo dimunuisce (bonus), altrimenti assume valore negativo (penalità).
long_green_penalty e balance_bonus sono eclusi dal calcolo differenziale.

# Riproduzione Risultati

Una volta installate le dipendenze si può procedere all'addestramento del modello che viene effettuato basandosi sui flussi generati dai dati estrapolati dal Junction Analytics di TOMTOM. Per 1_000_000 di timesteps il modello può impiegare fino a 2 ore per il training. 
Dopodiché si può testare il modello generando le statistiche di confronto con il semaforo a fasi fisse. Le fasi del semaforo sono già impostate nel file tls.tll.xml, pertanto nel caso in cui si volgia modificare le fasi fisse della baseline bisognerà agire modificando questo file. 
