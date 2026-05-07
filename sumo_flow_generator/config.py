# TomTom API Key
API_KEY = "jWfmaxSMOusB2IJMWCcOrfB32OHPn4cH"

# ID del cluster (corrisponde ai file CSV generati)
CLUSTER_ID = 0

# Fasce orarie da analizzare (formato HH:MM-HH:MM)
# Mattina, mezzogiorno, sera — le 3 ore di punta tipiche
TIME_SLOTS = [
    "07:00-10:00",
    "13:00-15:00",
    "18:00-21:00",
]

# Fattore di scala veicoli (1.0 = nessuna modifica, 0.5 = dimezza, 2.0 = raddoppia)
TRAFFIC_SCALE_FACTOR = 1.0

# Percorsi file input/output
EDGES_FILE          = "data/edges.csv"
FLOWS_REF_FILE      = f"data/traffic_flows_cluster_{CLUSTER_ID}_ref.csv"
FLOWS_COORDS_FILE   = f"data/output_flows_with_coordinates_cluster_{CLUSTER_ID}.csv"
OUTPUT_DIR          = "output"

# Parametri TomTom Flow API
TOMTOM_FLOW_ZOOM = 18          # Zoom level: più alto = più preciso (10-22)
TOMTOM_FLOW_UNIT = "KMPH"      # Unità velocità: KMPH o MPH

# Parametri simulazione SUMO
# begin/end in secondi dall'inizio della simulazione
SUMO_TIME_WINDOWS = {
    "07:00-10:00": {"begin": 0,    "end": 10800},   # 3 ore = 10800s
    "13:00-15:00": {"begin": 0,    "end": 7200},    # 2 ore = 7200s
    "18:00-21:00": {"begin": 0,    "end": 10800},   # 3 ore = 10800s
}

# Densità massima veicoli per corsia (veicoli/km) — usata da Greenshields
# Valore tipico urbano: 1 veicolo ogni 7 metri = ~143 veicoli/km
K_JAM = 143  # veicoli/km per corsia