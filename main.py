from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from baseline_builder import BaselineBuilder
from population_projector import poblacion_proyectada
from interdiction_optimizer import InterdictionOptimizer
import networkx as nw

ESCENARIO_POBLACION = "baja"   # "baja" | "central" | "alta"
ANIO_OBJETIVO       = 2025

PRESUPUESTO_INTERDICCION = 2      # máximo de aristas a cortar por escenario
TOP_N_CANDIDATOS         = 40     # tamaño del pool de aristas candidatas
POBLACION_GA             = 10
GENERACIONES_GA          = 10

Grafo_Comayagua = GraphConstructor("Comayagua,Honduras")
Grafo_sucio = GraphConstructor("Comayagua,Honduras")
graph = Grafo_Comayagua.construct_graph()
dirty_graph = Grafo_sucio.construct_dirt_graph()

Grafo_sucio.print_graph_info()
Grafo_sucio.visualize_graph()
print(nw.is_strongly_connected(dirty_graph))

#Grafo_Comayagua.print_graph_info()
#Grafo_Comayagua.visualize_graph()
#print(nw.is_strongly_connected(graph))

Grafo_Comayagua.clean_graph()

Grafo_Comayagua.print_graph_info()
print(nw.is_strongly_connected(graph))
Grafo_Comayagua.visualize_graph()

pop_por_barrio, total_pop = poblacion_proyectada(
    escenario=ESCENARIO_POBLACION, anio_objetivo=ANIO_OBJETIVO,
)
print(f"Proyección demográfica {ANIO_OBJETIVO} ({ESCENARIO_POBLACION}): "
      f"{total_pop:,} hab. en {len(pop_por_barrio)} zonas")

od_builder = ODMatrixBuilder(
    graph,
    population_by_zone=pop_por_barrio,
    total_population=total_pop,
)
od_builder.build_od_matrix()
od_builder.print_summary()
od_builder.plot_zone_map()

baseline = BaselineBuilder(graph, od_builder.get_matrix(), od_builder.nodes, paths=od_builder.get_paths())
baseline.build_routes()
baseline.print_summary()
baseline.animate_flow(n_frames=60, interval=120)

optimizer = InterdictionOptimizer(
    graph,
    od_builder.get_matrix(),
    od_builder.get_travel_times_dataframe().values,
    od_builder.nodes,
    od_builder.get_paths(),
    baseline.edge_flows,
    budget=PRESUPUESTO_INTERDICCION,
    top_n=TOP_N_CANDIDATOS,
    pop_size=POBLACION_GA,
    n_gen=GENERACIONES_GA,
)
optimizer.run()
print(f"\nHipervolumen del frente de Pareto: {optimizer.hypervolume():.2f}")

print("\nTop 10 aristas críticas (frecuencia en el frente de Pareto):")
for edge, count in optimizer.edge_criticality_ranking()[:10]:
    print(f"  {edge} -> en {count} soluciones no dominadas")

# optimizer.animate_evolution(interval=300)  # replay a otra velocidad / exportar a GIF