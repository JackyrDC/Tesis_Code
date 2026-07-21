import os

import networkx as nw

from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from baseline_builder import BaselineBuilder
from population_projector import poblacion_proyectada
from interdiction_optimizer import InterdictionOptimizer
from random_failures import run_random_failure_analysis, directed_optimum_by_size

ESCENARIO_POBLACION = "central"   # "baja" | "central" | "alta"
ANIO_OBJETIVO       = 2025

SEMILLA = 49                      # semilla única de toda la corrida (GA, muestreos, Monte Carlo)

PRESUPUESTO_INTERDICCION = 5      # máximo de aristas a cortar por escenario
TOP_N_CANDIDATOS         = 100     # tamaño del pool de aristas candidatas
POBLACION_GA             = 200
GENERACIONES_GA          = 50
PROCESOS_GA              = None   # None = automático (CPUs - 1); 1 = sin multiprocessing
REFRESCO_MAPA_VIVO       = 5      # redibujar el mapa del NSGA-II cada N generaciones
MUESTRAS_FALLOS_ALEAT    = 50    # muestras Monte Carlo para el contraste dirigido vs aleatorio

MODAL_SPLIT       = 0.33   # fracción de viajes O-D en vehículo privado
ESCALA_CAPACIDAD  = 1.0    # calibración de capacidad (ver §calibración); sube/baja el v/c
FW_MAX_ITER       = 50     # iteraciones máximas de Frank-Wolfe (equilibrio)

# Caché local del grafo: congela los datos de OSM (reproducibilidad) y evita
# re-descargar en cada corrida. Borrar los .graphml para forzar re-descarga.
_CACHE_DIR        = os.path.join(os.path.dirname(__file__), "cache")
GRAFO_DRIVE_CACHE = os.path.join(_CACHE_DIR, "comayagua_drive.graphml")
GRAFO_ALL_CACHE   = os.path.join(_CACHE_DIR, "comayagua_all.graphml")


def main():
    Grafo_Comayagua = GraphConstructor("Comayagua,Honduras")
    graph = Grafo_Comayagua.construct_graph_cached(GRAFO_DRIVE_CACHE, network_type='drive')

    # Grafo "sucio" (network_type='all'): solo para inspección visual.
    # Se libera de inmediato — es más grande que el grafo 'drive' y
    # mantenerlo vivo duplica el consumo de RAM del resto del pipeline.
    Grafo_sucio = GraphConstructor("Comayagua,Honduras")
    dirty_graph = Grafo_sucio.construct_graph_cached(GRAFO_ALL_CACHE, network_type='all')
    Grafo_sucio.print_graph_info()
    Grafo_sucio.visualize_graph()
    print(nw.is_strongly_connected(dirty_graph))
    del dirty_graph, Grafo_sucio

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

    # od_builder.graph ya tiene speed_kph/travel_time garantizados
    graph = od_builder.graph
    # POIs con nombre para anotar los mapas de baseline e interdicción
    pois = od_builder.get_pois()

    baseline = BaselineBuilder(
        graph, od_builder.get_matrix(), od_builder.nodes,
        paths=od_builder.get_paths(),
        modal_split=MODAL_SPLIT,
        capacity_scale=ESCALA_CAPACIDAD,
        fw_max_iter=FW_MAX_ITER,
        pois=pois,
        seed=SEMILLA,
    )

    # Línea base canónica: equilibrio de usuario (Wardrop) por Frank-Wolfe.
    baseline.build_equilibrium()
    baseline.print_summary()

    # Dos animaciones: convergencia del equilibrio + acumulación muestreada.
    baseline.animate_convergence(interval=400)
    baseline.build_routes()
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
        n_jobs=PROCESOS_GA,
        live_update_every=REFRESCO_MAPA_VIVO,
        pois=pois,
        seed=SEMILLA,
    )
    optimizer.run()
    print(f"\nHipervolumen del frente de Pareto: {optimizer.hypervolume():.2f}")

    print("\nTop 10 aristas críticas (frecuencia en el frente de Pareto):")
    for edge, count in optimizer.edge_criticality_ranking()[:10]:
        print(f"  {edge} -> en {count} soluciones no dominadas")

    # Persistir resultados ANTES del explorador interactivo, para no depender
    # de que la ventana se cierre correctamente.
    run_dir = optimizer.save_run(params={
        "escenario_poblacion": ESCENARIO_POBLACION,
        "anio_objetivo": ANIO_OBJETIVO,
        "presupuesto_interdiccion": PRESUPUESTO_INTERDICCION,
        "top_n_candidatos": TOP_N_CANDIDATOS,
        "poblacion_ga": POBLACION_GA,
        "generaciones_ga": GENERACIONES_GA,
        "semilla": SEMILLA,
        "modal_split": MODAL_SPLIT,
        "escala_capacidad": ESCALA_CAPACIDAD,
        "fw_max_iter": FW_MAX_ITER,
        "poblacion_total": total_pop,
    })

    # Fase 4 — contraste fallos dirigidos (NSGA-II) vs. aleatorios (Monte Carlo):
    # cuánto más daño causa un ataque dirigido que una falla fortuita del mismo
    # tamaño. Se guarda junto al resto de resultados de la corrida.
    directed = directed_optimum_by_size(optimizer.result)
    if directed:
        run_random_failure_analysis(
            graph,
            od_builder.get_matrix(),
            od_builder.get_travel_times_dataframe().values,
            od_builder.nodes,
            od_builder.get_paths(),
            cut_sizes=sorted(directed),
            n_samples=MUESTRAS_FALLOS_ALEAT,
            seed=SEMILLA,
            directed_by_size=directed,
            save_path=os.path.join(run_dir, "random_failures.csv") if run_dir else None,
        )

    # Explorador post-ejecución: mismo mapa con slider para navegar el frente
    # generación a generación (bloquea hasta cerrar la ventana).
    optimizer.explore_history()

    # optimizer.animate_evolution(interval=300)  # replay a otra velocidad / exportar a GIF


# Guard OBLIGATORIO: el NSGA-II usa multiprocessing y en Windows cada worker
# re-importa este módulo al arrancar; sin el guard, cada worker relanzaría
# el pipeline completo (descarga del grafo, matriz O-D, ...) en cascada.
if __name__ == "__main__":
    main()
