#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import defaultdict
from datetime import datetime
import os

# ----------------------- MAPPING DIRIZIONI -> EDGES SUMO -----------------------
tomtom_dir_to_in_edge = {
    "NORTH": "sud_in",
    "SOUTH": "nord_in",
    "EAST":  "ovest_in",
    "WEST":  "est_in"
}

tomtom_dir_to_out_edge = {
    "NORTH": "nord_out",
    "SOUTH": "sud_out",
    "EAST":  "est_out",
    "WEST":  "ovest_out"
}

# ----------------------- FASCE ORARIE UTC -----------------------
TIME_SLOTS = {
    "6_7":   (6, 7),   # 6:00-6:59:59
    "12_13": (12, 13),
    "16_17": (16, 17)
}

def get_time_slot(dt):
    """Restituisce la chiave della fascia oraria (es. '6_7') se l'ora ricade in una fascia, altrimenti None."""
    hour = dt.hour
    for slot, (start, end) in TIME_SLOTS.items():
        if start <= hour < end:
            return slot
    return None

def is_monday_or_tuesday(dt):
    return dt.weekday() in (0, 1)

def load_mappings(definition_path):
    approach_map = {}
    exit_map = {}
    with open(definition_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            typ = row['type']
            dir_raw = row['direction'].strip().upper()
            obj_id = int(row['id'])
            if typ == 'APPROACH':
                in_edge = tomtom_dir_to_in_edge.get(dir_raw)
                if in_edge:
                    approach_map[obj_id] = in_edge
            elif typ == 'EXIT':
                out_edge = tomtom_dir_to_out_edge.get(dir_raw)
                if out_edge:
                    exit_map[obj_id] = out_edge
    return approach_map, exit_map

def read_approaches_by_slot(approaches_path):
    """Restituisce un dizionario: slot -> {approach_id: volume_orario_medio}"""
    slot_volumes = {slot: defaultdict(list) for slot in TIME_SLOTS}
    with open(approaches_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dt = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
            except:
                continue
            if not is_monday_or_tuesday(dt):
                continue
            slot = get_time_slot(dt)
            if slot is None:
                continue
            approach_id = int(row['approachId'])
            try:
                vol = float(row['volumePerHour'])
            except:
                continue
            slot_volumes[slot][approach_id].append(vol)
    # Calcola media per ogni slot e approach
    slot_avg = {}
    for slot, app_dict in slot_volumes.items():
        avg_dict = {}
        for aid, vlist in app_dict.items():
            if vlist:
                avg_dict[aid] = sum(vlist) / len(vlist)
        slot_avg[slot] = avg_dict
    return slot_avg

def read_turn_ratios_by_slot(ratios_path):
    """Restituisce slot -> {approach_id: [(exit_id, prob_media), ...]}"""
    slot_ratios = {slot: defaultdict(lambda: defaultdict(float)) for slot in TIME_SLOTS}
    slot_counts = {slot: defaultdict(lambda: defaultdict(int)) for slot in TIME_SLOTS}
    with open(ratios_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dt = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
            except:
                continue
            if not is_monday_or_tuesday(dt):
                continue
            slot = get_time_slot(dt)
            if slot is None:
                continue
            approach_id = int(row['approachId'])
            exit_id = int(row['exitId'])
            try:
                perc = float(row['ratioPercent'])
            except:
                continue
            slot_ratios[slot][approach_id][exit_id] += perc
            slot_counts[slot][approach_id][exit_id] += 1
    # Media e normalizzazione a somma 1
    slot_result = {}
    for slot, app_dict in slot_ratios.items():
        result_app = {}
        for aid, exit_dict in app_dict.items():
            mean_list = []
            for eid, total in exit_dict.items():
                cnt = slot_counts[slot][aid][eid]
                if cnt > 0:
                    mean_perc = total / cnt
                    mean_list.append((eid, mean_perc))
            # Normalizza
            total_perc = sum(p for _, p in mean_list)
            if total_perc > 0:
                norm = [(eid, p / total_perc) for eid, p in mean_list]
                result_app[aid] = norm
        slot_result[slot] = result_app
    return slot_result

def generate_rou_xml(volumes, turn_ratios, approach_map, exit_map, output_path, sim_duration=3600):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="carro" acceleration="2.6" deceleration="4.5" length="5" maxSpeed="13.89"/>\n\n')
        
        # Aggrega volumi per incoming edge SUMO
        edge_volumes = defaultdict(float)
        for aid, vol in volumes.items():
            in_edge = approach_map.get(aid)
            if in_edge:
                edge_volumes[in_edge] += vol
        
        # Aggrega svolte per incoming edge
        edge_turns = defaultdict(list)
        for aid, turns in turn_ratios.items():
            in_edge = approach_map.get(aid)
            if not in_edge:
                continue
            for eid, prob in turns:
                out_edge = exit_map.get(eid)
                if out_edge:
                    edge_turns[in_edge].append((out_edge, prob))
        
        ordered_in_edges = ['est_in', 'ovest_in', 'nord_in', 'sud_in']
        for in_edge in ordered_in_edges:
            if in_edge not in edge_volumes or edge_volumes[in_edge] == 0:
                continue
            total_veh = edge_volumes[in_edge]
            turns = edge_turns.get(in_edge, [])
            if not turns:
                continue
            dist_id = f"svolte_{in_edge[:-3]}"
            f.write(f'    <routeDistribution id="{dist_id}">\n')
            for out_edge, prob in turns:
                f.write(f'        <route edges="{in_edge} {out_edge}" cost="1" probability="{prob:.4f}"/>\n')
            f.write('    </routeDistribution>\n\n')
            f.write(f'    <flow id="flow_{in_edge[:-3]}" begin="0" end="{sim_duration}" vehsPerHour="{total_veh:.1f}" type="carro" route="{dist_id}"/>\n\n')
        
        f.write('</routes>\n')

def main():
    definition_file = "2026_05_09_11_16_20_definition.csv"
    approaches_file = "approaches.csv"
    turn_ratios_file = "turn_ratios.csv"
    
    for f in [definition_file, approaches_file, turn_ratios_file]:
        if not os.path.exists(f):
            print(f"Errore: file {f} non trovato.")
            return
    
    print("Caricamento mapping approcci/uscite...")
    approach_map, exit_map = load_mappings(definition_file)
    print(f"  Mappati {len(approach_map)} approcci, {len(exit_map)} uscite.")
    
    print("Lettura volumi per fascia oraria (lunedì/martedì)...")
    slot_volumes = read_approaches_by_slot(approaches_file)
    
    print("Lettura turn ratios per fascia oraria...")
    slot_turn_ratios = read_turn_ratios_by_slot(turn_ratios_file)
    
    for slot in TIME_SLOTS:
        volumes = slot_volumes.get(slot, {})
        turn_ratios = slot_turn_ratios.get(slot, {})
        if not volumes or not turn_ratios:
            print(f"Attenzione: nessun dato per fascia {slot}, salto.")
            continue
        output_file = f"flussi_calibrati_{slot}.rou.xml"
        print(f"Generazione {output_file}...")
        generate_rou_xml(volumes, turn_ratios, approach_map, exit_map, output_file)
        print(f"  Creato {output_file}")
    
    print("Fatto.")

if __name__ == "__main__":
    main()