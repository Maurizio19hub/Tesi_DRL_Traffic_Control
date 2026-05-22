output_route_file = "flussi_calibrati.rou.xml"

# Definizione corretta utilizzando <routeDistribution> per evitare conflitti in SUMO
routes_xml = """<routes>
    <vType id="carro" acceleration="2.6" deceleration="4.5" length="5" maxSpeed="13.89"/>

    <routeDistribution id="svolte_est">
        <route edges="est_in ovest_out" cost="1" probability="0.70"/> <route edges="est_in nord_out" cost="1" probability="0.15"/>  <route edges="est_in sud_out" cost="1" probability="0.15"/>   </routeDistribution>

    <routeDistribution id="svolte_ovest">
        <route edges="ovest_in est_out" cost="1" probability="0.70"/>  <route edges="ovest_in sud_out" cost="1" probability="0.15"/>   <route edges="ovest_in nord_out" cost="1" probability="0.15"/>  </routeDistribution>

    <routeDistribution id="svolte_nord">
        <route edges="nord_in sud_out" cost="1" probability="0.65"/>   <route edges="nord_in ovest_out" cost="1" probability="0.25"/> <route edges="nord_in est_out" cost="1" probability="0.10"/>   </routeDistribution>

    <routeDistribution id="svolte_sud">
        <route edges="sud_in nord_out" cost="1" probability="0.65"/>   <route edges="sud_in est_out" cost="1" probability="0.25"/>   <route edges="sud_in ovest_out" cost="1" probability="0.10"/>  </routeDistribution>


    <flow id="flow_est" begin="0" end="3600" vehsPerHour="450" type="carro" route="svolte_est"/>
    
    <flow id="flow_ovest" begin="0" end="3600" vehsPerHour="450" type="carro" route="svolte_ovest"/>
    
    <flow id="flow_nord" begin="0" end="3600" vehsPerHour="220" type="carro" route="svolte_nord"/>
    
    <flow id="flow_sud" begin="0" end="3600" vehsPerHour="220" type="carro" route="svolte_sud"/>

</routes>
"""

with open(output_route_file, "w") as f:
    f.write(routes_xml)

print(f"File di rotte corretto e calibrato generato con successo: {output_route_file}")