"""
Línea base greedy sobre una corrida YA guardada del NSGA-II, sin
re-ejecutar el optimizador.

Reconstruye de forma determinista la parte inicial del pipeline (grafo
desde caché -> proyección demográfica -> matriz O-D -> equilibrio de
Wardrop) usando los parámetros registrados en el metadata.json de la
corrida, de modo que el pool de candidatas es EXACTAMENTE el que vio el
NSGA-II de esa corrida. El óptimo dirigido por tamaño de corte no se
recalcula: se lee de la columna `dirigido` de random_failures.csv.

La tabla comparativa queda en <run>/greedy_baseline.csv, junto al resto
de resultados de la corrida.

Uso:
    python run_greedy.py results/run_20260717_172220
    python run_greedy.py                # usa RUN_DIR de abajo
"""

import csv
import json
import os
import sys

# La consola de Windows usa cp1252 y revienta al imprimir caracteres como
# '→' (flecha) que emiten los modulos del pipeline. Forzamos UTF-8 en la
# salida para que el proceso no aborte con UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from baseline_builder import BaselineBuilder
from population_projector import poblacion_proyectada
from interdiction_optimizer import build_candidate_pool
from greedy_baseline import run_greedy_baseline_analysis

# Corrida a analizar: cambiar aquí, o pasarla como primer argumento.
RUN_DIR = os.path.join(os.path.dirname(__file__), "results", "run_20260717_172220")

_CACHE_DIR        = os.path.join(os.path.dirname(__file__), "cache")
GRAFO_DRIVE_CACHE = os.path.join(_CACHE_DIR, "comayagua_drive.graphml")


def cargar_parametros(run_dir: str) -> dict:
    """Parámetros con los que se ejecutó la corrida (metadata.json)."""
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return meta["parametros"]


def cargar_dirigido(run_dir: str) -> dict[int, float]:
    """
    {k: ΔTTV óptimo dirigido} desde random_failures.csv — el mismo dict que
    produce `directed_optimum_by_size`, pero sin re-correr el NSGA-II.
    """
    directed: dict[int, float] = {}
    with open(os.path.join(run_dir, "random_failures.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dirigido"):
                directed[int(row["num_cortes"])] = float(row["dirigido"])
    return directed


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR
    if not os.path.isdir(run_dir):
        sys.exit(f"No existe la carpeta de corrida: {run_dir}")

    params = cargar_parametros(run_dir)
    directed = cargar_dirigido(run_dir)
    if not directed:
        sys.exit(f"random_failures.csv sin columna 'dirigido' en {run_dir}")

    cut_sizes = sorted(directed)
    print(f"Corrida: {os.path.basename(run_dir)}")
    print(f"  escenario={params['escenario_poblacion']}  "
          f"anio={params['anio_objetivo']}  top_n={params['top_n_candidatos']}  "
          f"modal_split={params['modal_split']}  fw_max_iter={params['fw_max_iter']}")
    print(f"  tamanos de corte a evaluar: {cut_sizes}")

    # ── Reconstrucción determinista del pipeline hasta el equilibrio ─────
    # Mismos pasos que main.py, sin visualizaciones ni grafo 'all'.
    constructor = GraphConstructor("Comayagua,Honduras")
    constructor.construct_graph_cached(GRAFO_DRIVE_CACHE, network_type="drive")
    constructor.clean_graph()

    pop_por_barrio, total_pop = poblacion_proyectada(
        escenario=params["escenario_poblacion"],
        anio_objetivo=params["anio_objetivo"],
    )

    od_builder = ODMatrixBuilder(
        constructor.graph,
        population_by_zone=pop_por_barrio,
        total_population=total_pop,
    )
    od_builder.build_od_matrix()

    # od_builder.graph ya tiene speed_kph/travel_time garantizados
    graph = od_builder.graph

    baseline = BaselineBuilder(
        graph, od_builder.get_matrix(), od_builder.nodes,
        paths=od_builder.get_paths(),
        modal_split=params["modal_split"],
        capacity_scale=params["escala_capacidad"],
        fw_max_iter=params["fw_max_iter"],
    )
    baseline.build_equilibrium()

    pool = build_candidate_pool(
        graph, baseline.edge_flows, top_n=params["top_n_candidatos"],
    )

    run_greedy_baseline_analysis(
        graph,
        od_builder.get_matrix(),
        od_builder.get_travel_times_dataframe().values,
        od_builder.nodes,
        od_builder.get_paths(),
        pool,
        cut_sizes=cut_sizes,
        directed_by_size=directed,
        save_path=os.path.join(run_dir, "greedy_baseline.csv"),
    )


if __name__ == "__main__":
    main()
