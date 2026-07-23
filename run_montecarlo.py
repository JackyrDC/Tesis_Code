"""
Fallos aleatorios (Monte Carlo) sobre una corrida YA guardada del NSGA-II,
sin re-ejecutar el optimizador. Análogo a `run_greedy.py`.

Reconstruye de forma determinista la parte inicial del pipeline (grafo desde
caché -> proyección demográfica -> matriz O-D -> equilibrio de Wardrop) con
los parámetros del metadata.json de la corrida, para que el universo de
aristas y los tiempos de viaje sean EXACTAMENTE los que vio esa corrida.

El óptimo dirigido por tamaño de corte no requiere re-correr el NSGA-II: se
deriva del mejor ΔTTV por num_cortes en la ÚLTIMA generación de
`pareto_front_by_generation.csv` (equivalente a `directed_optimum_by_size`
sobre `optimizer.result`, pero leído desde disco).

La tabla queda en <run>/random_failures.csv, junto al resto de resultados.

Uso:
    python run_montecarlo.py results/run_20260721_193208
    python run_montecarlo.py results/run_20260721_193208 --n-samples 50
"""

import argparse
import csv
import json
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from baseline_builder import BaselineBuilder
from population_projector import poblacion_proyectada
from random_failures import run_random_failure_analysis

_CACHE_DIR        = os.path.join(os.path.dirname(__file__), "cache")
GRAFO_DRIVE_CACHE = os.path.join(_CACHE_DIR, "comayagua_drive.graphml")


def cargar_parametros(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return meta["parametros"]


def cargar_dirigido_desde_pareto(run_dir: str) -> dict[int, float]:
    """
    {k: ΔTTV óptimo dirigido} tomando el máximo delta_ttv por num_cortes en
    la última generación de pareto_front_by_generation.csv.
    """
    path = os.path.join(run_dir, "pareto_front_by_generation.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {}
    ultima_gen = max(int(r["generacion"]) for r in rows)
    directed: dict[int, float] = {}
    for r in rows:
        if int(r["generacion"]) != ultima_gen:
            continue
        k = int(r["num_cortes"])
        if k <= 0:
            continue
        delta = float(r["delta_ttv"])
        directed[k] = max(directed.get(k, 0.0), delta)
    return directed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--n-samples", type=int, default=50,
                         help="muestras Monte Carlo por tamaño de corte (default: 50, "
                              "para ser comparable con las corridas limpias existentes)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        sys.exit(f"No existe la carpeta de corrida: {run_dir}")

    params = cargar_parametros(run_dir)
    directed = cargar_dirigido_desde_pareto(run_dir)
    if not directed:
        sys.exit(f"No se pudo derivar el óptimo dirigido desde pareto_front_by_generation.csv en {run_dir}")

    cut_sizes = sorted(directed)
    print(f"Corrida: {os.path.basename(run_dir)}")
    print(f"  escenario={params['escenario_poblacion']}  anio={params['anio_objetivo']}  "
          f"modal_split={params['modal_split']}  fw_max_iter={params['fw_max_iter']}  "
          f"semilla={params['semilla']}")
    print(f"  tamanos de corte a evaluar: {cut_sizes}  n_samples={args.n_samples}")

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

    graph = od_builder.graph

    baseline = BaselineBuilder(
        graph, od_builder.get_matrix(), od_builder.nodes,
        paths=od_builder.get_paths(),
        modal_split=params["modal_split"],
        capacity_scale=params["escala_capacidad"],
        fw_max_iter=params["fw_max_iter"],
    )
    baseline.build_equilibrium()

    run_random_failure_analysis(
        graph,
        od_builder.get_matrix(),
        od_builder.get_travel_times_dataframe().values,
        od_builder.nodes,
        od_builder.get_paths(),
        cut_sizes=cut_sizes,
        n_samples=args.n_samples,
        seed=params["semilla"],
        directed_by_size=directed,
        save_path=os.path.join(run_dir, "random_failures.csv"),
    )


if __name__ == "__main__":
    main()
