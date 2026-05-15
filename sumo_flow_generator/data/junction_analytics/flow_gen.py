import pandas as pd
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os

# ── File di input/output ──────────────────────────────────────
FILE_APPROACHES  = "data_1/csv/live-data/approaches.csv"
FILE_TURN_RATIOS = "data_1/csv/live-data/turn_ratios.csv"
OUTPUT_ROU       = "output/incrocio3.rou.xml"

# ── Mappe ID TomTom → edge SUMO ──────────────────────────────
APPROACH_MAPPING = {
    257100653:   'est_in',
    -1881102440: 'ovest_in',
    625156923:   'sud_in',
    1676040914:  'nord_in',
}
EXIT_MAPPING = {
    537023766:   'ovest_out',
    -1432146481: 'est_out',
    -359437628:  'nord_out',
    -1635665529: 'sud_out',
}

# ── U-turn non permessi ───────────────────────────────────────
UTURN = {
    ('est_in',   'est_out'),
    ('ovest_in', 'ovest_out'),
    ('nord_in',  'nord_out'),
    ('sud_in',   'sud_out'),
}

# ── Fasce orarie ─────────────────────────────────────────────
FASCE_ORARIE = {
    "Mattina_Picco":      {"ora_utc_csv": 6,  "inizio_sumo": 0,    "fine_sumo": 3600},
    "Pomeriggio_Morbida": {"ora_utc_csv": 12, "inizio_sumo": 3600, "fine_sumo": 7200},
    "Sera_Picco":         {"ora_utc_csv": 16, "inizio_sumo": 7200, "fine_sumo": 10800},
}

# ── Tipo veicolo ─────────────────────────────────────────────
VEHICLE_TYPE = {
    "id":       "auto",
    "vClass":   "passenger",
    "length":   "4.5",
    "accel":    "2.6",
    "decel":    "4.5",
    "sigma":    "0.5",
    "maxSpeed": "13.89",
}


def elabora_dati_traffico():
    print("Lettura e filtraggio dati TomTom...")

    df_app  = pd.read_csv(FILE_APPROACHES)
    df_turn = pd.read_csv(FILE_TURN_RATIOS)

    df_app['time']  = pd.to_datetime(df_app['time'])
    df_turn['time'] = pd.to_datetime(df_turn['time'])

    # Filtra solo lunedì (0) e martedì (1)
    df_app  = df_app[df_app['time'].dt.dayofweek.isin([0, 1])].copy()
    df_turn = df_turn[df_turn['time'].dt.dayofweek.isin([0, 1])].copy()

    df_app['hour_utc']  = df_app['time'].dt.hour
    df_turn['hour_utc'] = df_turn['time'].dt.hour

    ore_utc_target = [info["ora_utc_csv"] for info in FASCE_ORARIE.values()]
    df_app  = df_app[df_app['hour_utc'].isin(ore_utc_target)].copy()
    df_turn = df_turn[df_turn['hour_utc'].isin(ore_utc_target)].copy()

    df_app['edge_in']   = df_app['approachId'].map(APPROACH_MAPPING)
    df_turn['edge_in']  = df_turn['approachId'].map(APPROACH_MAPPING)
    df_turn['edge_out'] = df_turn['exitId'].map(EXIT_MAPPING)

    df_app  = df_app.dropna(subset=['edge_in'])
    df_turn = df_turn.dropna(subset=['edge_in', 'edge_out'])

    # Rimuovi U-turn
    df_turn = df_turn[
        ~df_turn.apply(lambda r: (r['edge_in'], r['edge_out']) in UTURN, axis=1)
    ]

    print("Calcolo flussi e frazioni di svolta medie (lunedì + martedì)...")
    flussi_medi = (
        df_app.groupby(['hour_utc', 'edge_in'])['volumePerHour']
        .mean()
        .reset_index()
    )
    svolte_medie = (
        df_turn.groupby(['hour_utc', 'edge_in', 'edge_out'])['ratioPercent']
        .mean()
        .reset_index()
    )

    return flussi_medi, svolte_medie


def genera_file_xml(flussi, svolte):
    os.makedirs(os.path.dirname(OUTPUT_ROU), exist_ok=True)
    print(f"Scrittura file XML: {OUTPUT_ROU}...")

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    vtype = ET.SubElement(root, "vType")
    for k, v in VEHICLE_TYPE.items():
        vtype.set(k, v)

    flow_counter = 0

    for nome_fascia, info in FASCE_ORARIE.items():
        ora_utc = info["ora_utc_csv"]
        inizio  = info["inizio_sumo"]
        fine    = info["fine_sumo"]

        root.append(ET.Comment(f" FASCIA: {nome_fascia} (UTC {ora_utc}:00) "))

        sub_flussi = flussi[flussi['hour_utc'] == ora_utc]

        for _, row_flusso in sub_flussi.iterrows():
            edge_in      = row_flusso['edge_in']
            veh_per_hour = row_flusso['volumePerHour']

            if veh_per_hour <= 0:
                continue

            sub_svolte = svolte[
                (svolte['hour_utc'] == ora_utc) &
                (svolte['edge_in']  == edge_in)
            ]
            if sub_svolte.empty:
                continue

            # Normalizza dopo rimozione U-turn
            tot_percent = sub_svolte['ratioPercent'].sum()
            if tot_percent <= 0:
                continue

            # Un flow separato per ogni coppia O/D con probability scalata per la svolta
            for _, row_svolta in sub_svolte.iterrows():
                edge_out = row_svolta['edge_out']
                frazione = row_svolta['ratioPercent'] / tot_percent

                # Probability al secondo scalata per la frazione di svolta
                prob = round((veh_per_hour / 3600.0) * frazione, 6)
                if prob <= 0:
                    continue

                flow = ET.SubElement(root, "flow")
                flow.set("id",          f"flow_{flow_counter}")
                flow.set("type",        "auto")
                flow.set("from",        edge_in)
                flow.set("to",          edge_out)
                flow.set("begin",       str(inizio))
                flow.set("end",         str(fine))
                flow.set("probability", str(prob))
                flow.set("departLane",  "best")
                flow.set("departSpeed", "random")

                print(f"  {nome_fascia} | {edge_in} → {edge_out}: "
                      f"{veh_per_hour:.0f} veh/h × {frazione:.2%} = {prob:.5f} veh/s")
                flow_counter += 1

    xml_string = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="    ", encoding="utf-8")

    with open(OUTPUT_ROU, "wb") as f:
        f.write(pretty_xml)

    print(f"\n✓ File generato: {OUTPUT_ROU} ({flow_counter} flow totali)")


if __name__ == "__main__":
    flussi, svolte = elabora_dati_traffico()
    genera_file_xml(flussi, svolte)