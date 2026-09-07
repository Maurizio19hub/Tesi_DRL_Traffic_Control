# DRL-Traffic-Intersection

## 📌 Panoramica del progetto

Questo progetto applica il **Deep Reinforcement Learning (DRL)** con l'algoritmo **PPO** per ottimizzare la gestione semaforica di un incrocio simulato con **SUMO** (Simulation of Urban MObility).  
L'obiettivo è ridurre i tempi di attesa, le code, le emissioni e migliorare il flusso veicolare, confrontando le prestazioni dell'agente PPO con due baseline a fasi fisse.

---

## 🛠️ Versioni software usate per lo sviluppo

- **Python**: `3.14.3`
- **SUMO**: `Eclipse SUMO v1.26.0+0485-c820d16e9df`
- **Sistema operativo**: `Linux (Fedora 43)`

---

## 📁 Struttura della cartella

📁 DRL-Traffic-Intersection/                            # Root del progetto
│
├── 📁 csv_generati/                                    # Dati di valutazione (generati da generate_csv.py)
│   ├── 📁 baseline/                                    # Baseline standard (semaforo non ottimizzato)
│   │   ├── 📄 metrics_baseline.csv                     # Metriche aggregate su 20 seed
│   │   └── 📄 timeseries_baseline.npz                  # Serie temporali step-by-step
│   ├── 📁 baseline_opt/                                # Baseline ottimizzata (semaforo con fasi ottimizzate)
│   │   ├── 📄 metrics_baseline.csv
│   │   └── 📄 timeseries_baseline.npz
│   ├── 📁 PPO/                                         # Modello PPO energy + pedwaiting (senza collisioni)
│   │   ├── 📄 metrics_ppo.csv                          # Metriche del modello PPO
│   │   └── 📄 timeseries_ppo.npz                       # Serie temporali del modello PPO
│   └── 📁 PPO-collisions/                              # Modello PPO energy + pedwaiting + collisioni
│       ├── 📄 metrics_ppo.csv
│       └── 📄 timeseries_ppo.npz
│
├── 📁 models/                                          # Modelli addestrati e checkpoint
│   ├── 📁 t20energy-collisions_f2/                     # Modello con solo energia + collisioni
│   │   ├── 📄 monitor.csv                              # Log reward per episodio
│   │   ├── 📄 ppo_semaforo.zip                         # Modello PPO finale
│   │   ├── 📄 training_curve.png                       # Curva di training
│   │   └── 📄 vec_normalize_stats.pkl                  # Statistiche normalizzazione
│   ├── 📁 t20energy-pedwaiting-collisions_f2/          # Modello energia + pedoni + collisioni
│   │   ├── 📄 monitor.csv
│   │   ├── 📄 ppo_semaforo.zip
│   │   ├── 📄 training_curve.png
│   │   └── 📄 vec_normalize_stats.pkl
│   ├── 📁 t20energy-pedwaiting_f2/                     # Modello energia + pedoni (usato per reward)
│   │   ├── 📄 monitor.csv
│   │   ├── 📄 ppo_semaforo.zip
│   │   ├── 📄 training_curve.png
│   │   └── 📄 vec_normalize_stats.pkl
│   └── 📁 t20energy-pedwaiting_f2_trstats/             # Modello con metriche fisiche durante training
│       ├── 📄 monitor.csv                              
│       ├── 📄 ppo_semaforo.zip                        
│       ├── 📄 training_physical_metrics.csv            # Metriche fisiche (code, CO2, etc.) per episodio
│       ├── 📄 vec_normalize_stats.pkl                 
│       └── 📄 ppo_checkpoint_*_steps.zip               # 145 checkpoint salvati (da 3.6k a 1M steps)
│
├── 📁 new_sumo_flow_gen/                               # Generazione flussi veicolari calibrati
│   ├── 📄 flow_gen.py                                  # Script per generare flussi
│   ├── 📄 flussi_calibrati_12_13.rou.xml               # File flussi generato per ora 12-13
│   ├── 📄 approaches.csv                               # Approcci stradali (TOMTOM)
│   ├── 📄 turn_ratios.csv                              # Rapporti di svolta (TOMTOM)
│   └── 📄 2026_05_09_11_16_20_definition.csv           # Definizione flussi (TOMTOM)
│
├── 📁 risultati_finali/                                # Output finale dei grafici
│   ├── 📁 PPO_energy_pedwaiting/                       # Grafici per modello energy + pedwaiting
│   │   ├── 📁 training/                                # Metriche fisiche durante training
│   │   │   └── 📄 training_metrics_trend.png           # Andamento metriche (media mobile 20 ep)
│   │   ├── 📁 reward/                                  # Curva di reward
│   │   │   └── 📄 training_reward.png                  # Reward + media mobile 10 ep
│   │   ├── 📁 confronto_PPO_baseline/                  # Confronto con baseline standard
│   │   │   ├── 📄 bar_comparison_all.png               # Barre media ± std per tutte le metriche
│   │   │   ├── 📄 boxplot_comparison_all.png           # Boxplot distribuzione su seed
│   │   │   ├── 📄 violin_comparison_all.png            # Violin plot distribuzione
│   │   │   ├── 📄 timeseries_comparison.png            # Serie temporali (media ± std)
│   │   │   ├── 📄 improvement_summary.png              # Barre orizzontali miglioramenti
│   │   │   ├── 📄 final_summary_table.png              # Tabella riassuntiva come immagine
│   │   │   ├── 📄 final_summary_table.csv              # Tabella in formato CSV
│   │   │   ├── 📄 final_summary_table.tex              # Tabella in formato LaTeX
│   │   │   ├── 📄 improvement_summary.csv              # Dati miglioramenti percentuali
│   │   │   └── 📄 final_summary_table_compact.csv      # Tabella compatta CSV (media ± std)
│   │   └── 📁 confronto_PPO_baseline-opt/              # Confronto con baseline ottimizzata
│   │       └── (stessi file della cartella sopra)
│   │
│   └── 📁 PPO_energy_pedwaiting_collisions/            # Grafici per modello energy + pedwaiting + collisioni
│       ├── 📁 reward/                                  # Curva di reward
│       │   └── 📄 training_reward.png
│       ├── 📁 confronto_PPO_baseline/                  # Confronto con baseline standard
│       │   └── (stessi file della cartella sopra)
│       └── 📁 confronto_PPO_baseline-opt/              # Confronto con baseline ottimizzata
│           └── (stessi file della cartella sopra)
│
├── 📁 video_simulazione/                               # Video delle simulazioni
│   ├── 📄 video_simulazione_baseline-opt.mp4           # Video baseline ottimizzata
│   └── 📄 video_simulazione_PPO.mp4                    # Video con agente PPO
│
├── 📄 generazione_grafici_finali.py                    # Script unico per generare TUTTI i grafici finali
├── 📄 generate_csv.py                                  # Genera CSV e .npz per baseline e PPO (valutazione)
├── 📄 sumo_env.py                                      # Ambiente Gym per SUMO (senza collisioni)
├── 📄 sumo_env_collisions.py                           # Ambiente Gym per SUMO (con collisioni)
├── 📄 main.py                                          # Script principale training PPO
├── 📄 test.py                                          # Script di test ambiente SUMO
├── 📄 debug.py                                         # Script di simulazione con modello addestrato
├── 📄 baseline_diff.py                                 # Script di simulazione per baseline
│
├── 📄 incrocio3_300mf.net.xml                          # Rete stradale SUMO
├── 📄 simulazione.sumocfg                              # Configurazione simulazione SUMO
├── 📄 tls.tll.xml                                      # Configurazione semaforo
├── 📄 trips.trips.xml                                  # Viaggi generati
├── 📄 pedoni.rou.xml                                   # File pedoni generato (output randomTrips)
│
├── 📄 README.md                                        # Documentazione progetto
└── 📄 requirements.txt                                 # Dipendenze Python



---

## 📦 Dipendenze

### 🔹 Python Standard Library
- `csv`
- `collections` (`defaultdict`)
- `datetime`
- `os`
- `subprocess`

### 🔹 Scientific Computing
- `numpy`
- `pandas`
- `matplotlib` (con `matplotlib.gridspec`)
- `seaborn`

### 🔹 SUMO
- `traci`

### 🔹 Reinforcement Learning
- `gymnasium`
- `stable-baselines3` (con moduli: `common.monitor`, `common.vec_env`, `common.callbacks`, `common.evaluation`)

### 🔹 Ambiente locale
- `sumo_env` (modulo locale, contiene la classe `MyEnv`)

---

## 🚀 Comandi per la riproduzione dei risultati

### 1. Creazione dell'ambiente virtuale (Conda)
```bash
conda create --name DRL-Traffic-Intersection python=3.14.3
conda activate DRL-Traffic-Intersection
```

### 2. Installazione delle librerie
```bash
pip install -r requirements.txt
```

### 3. Avvio training (nel file main.py si può modficare cartella in cui salvare il nuovo modello)
```bash
python main.py
```

### 4. Comando testing (si può modificare per testare modello addestrato su più seed, usato in fase di sviluppo):
```bash
python debug.py
```

### 5. Comando testing (si può modificare per testare baseline su più seed, usato in fase di sviluppo):
```bash
python baseline_diff.py
```

### 6. Comando generazione metriche del testing tramite SEEDS = list(range(0, 100, 5)) (20 seed) e salvataggio in csv :
```bash
python generate_csv.py
```

### 7. Comando generazione grafici (confronto, reward, metriche training) :
```bash
python generazione_grafici_finali.py
```

## ⚙️ Note sul training e sulla struttura dei risultati

- Una volta installate le dipendenze, si può procedere all'addestramento del modello, che si basa sui flussi generati dai dati estratti dal **Junction Analytics di TOMTOM**.
- Per **1_000_000 di timesteps** il training può richiedere fino a **5 ore**.
- Dopo il training, è possibile testare il modello seguendo i comandi sopra.
- Le **fasi del semaforo** sono già impostate nel file `tls.tll.xml`. Per modificare le fasi fisse della **baseline**, è necessario agire su questo file.
- La cartella `risultati_finali/` è suddivisa per modello (`PPO_energy_pedwaiting` e `PPO_energy_pedwaiting_collisions`). Ciascun modello viene confrontato con entrambe le baseline (`baseline` e `baseline_opt`).

---

## 🧠 Funzione di Reward

La funzione `_get_reward(self)` calcola la ricompensa per l'agente in base ai dati raccolti dalla simulazione tramite **TraCI**.

### Variabili utilizzate

| Variabile | Descrizione |
| :--- | :--- |
| `total_waiting` | Tempo di attesa normalizzato (somma su tutte le strade) |
| `total_queue` | Lunghezza delle code normalizzata (somma su tutte le strade) |
| `energy_penalty` | Calcolata tramite `_get_energy_penalty(self)` — combina una penalità per veicoli fermi e una penalità per veicoli in movimento che diminuisce all'aumentare della velocità. Questo favorisce il mantenimento del flusso veicolare a velocità sostenute, agendo indirettamente su accelerazioni/decelerazioni e quindi su consumi ed emissioni. |
| `ped_norm` | Pedoni in attesa (normalizzato) |
| `long_green_penalty` | Penalità incrementale quando l'agente mantiene la stessa fase per più di 60 secondi (**esclusa** dalla parte differenziale) |
| `balance_bonus` | Penalità/bonus incrementale che premia l'agente se mantiene code e tempi di attesa sotto determinate soglie (**escluso** dalla parte differenziale) |

### Calcolo della Reward

La formula usata nel modello migliore (**senza collisioni**) è composta da:

1. **Parte differenziale**: il costo al passo corrente viene confrontato con quello al passo precedente. Il segno positivo indica una riduzione del costo (bonus), quello negativo un aumento (penalità).

   ```python
   current_cost = (total_queue * 2.5) + (2 * total_waiting) + (energy_penalty * 0.01) + (ped_norm * 0.2)
   ret = self.last_cost - current_cost
   ```

2. **Termini aggiuntivi (esclusi dal differenziale)**: in aggiunta per il calcolo della reward restituita in uscita si hanno ulteriori due termini, il primo somma il contributo del bonus (negativo o positivo) e il secondo un contributo che penalizza l'agente se mantiene per troppo tempo la fase semaforica.
    
    ```python
    return (ret + balance_bonus)*20 + long_green_penalty
    ```

I pesi sono stati scelti in base alla **magnitudo** di ciascuna componente e alle priorità dell'agente nella gestione dell'incrocio. Analizzando i valori raccolti durante il training, si sono scelti pesi che:

- danno maggiore importanza a code e tempi di attesa;
- mantengono uno scarto dimensionale tra le componenti per evitare sovrapposizione o annullamento del segnale nella reward.

> **Nota**: nel caso della reward **con collisioni**, viene aggiunta una penalità specifica. Tuttavia, questo modello risulta meno performante rispetto a quello senza collisioni, come si può osservare dalle tabelle di confronto nei grafici generati.

---

## 📊 Risultati attesi

I grafici generati in `risultati_finali/` permettono di valutare:

- **Andamento delle metriche fisiche** durante il training (code, attese, velocità, emissioni, collisioni, ecc.).
- **Curva di reward** con media mobile per monitorare la convergenza dell'agente.
- **Confronto diretto PPO vs Baseline** tramite:
  - Barre con media e deviazione standard
  - Boxplot e violin plot per la distribuzione su 20 seed
  - Serie temporali medie con banda di confidenza
  - Tabella riassuntiva con miglioramenti percentuali

---





