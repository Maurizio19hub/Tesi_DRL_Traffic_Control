#COMANDO CONVERSIONE
netconvert --osm-files map-bari5.osm \                                                                                           
--output-file incrocio6.net.xml \
--junctions.join \
--geometry.remove \
--no-turnarounds true

#GENERAZIONE DEI FILE .rou
python "$SUMO_HOME/tools/randomTrips.py" -n incrocio.net.xml -e 3600 -p 1.5 --route-file veicoli.rou.xml
python "$SUMO_HOME/tools/randomTrips.py" -n incrocio.net.xml -e 3600 -p 3.0 --persontrips --route-file pedoni.rou.xml

cercare comando TELEPORT
TOMTOM:
-tomTraffic00
-mrzcamposeo@gmail.com


#TITOLO TESI

1) DEEP REINFORCEMENT LEARNING PER IL CONTROLLO ADATTIVO DI UN INCROCIO SEMAFORICO URBANO: CASO DI STUDIO A BARI
2) SIMULAZIONE E CONTROLLO DI UN INCROCIO SEMAFORICO DI BARI MEDIANTE DEEP REINFORCEMENT LEARNING
3) CONTROLLO ADATTIVO DI UN INCROCIO SEMAFORICO A BARI CON OTTIMIZZAZIONE MULTIMODALE TRAMITE DEEP REINFORCEMENT LEARNING


Possibile funzione di reward per la collisione : 
num_collisions = len(traci.simulation.getCollisions())
collision_penalty = -100.0 if num_collisions > 0 else 0.0
