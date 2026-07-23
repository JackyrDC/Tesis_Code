"""
Auditoría: ¿hay otras zonas O-D cuyo nodo-centroide está tan mal conectado
como el de BARRIO LOS LIRIOS (grado bajo, posible extremo muerto), y si
las hay, su arista de salida aparece en el pool de candidatas por alto
flujo? Responde la pregunta de si un artefacto así sería "visible" en los
resultados actuales por demanda inflada, o si podría estar escondido.

Uso:
    python audit_zone_centroids.py
"""

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from baseline_builder import BaselineBuilder
from population_projector import poblacion_proyectada
from interdiction_optimizer import build_candidate_pool

GRADO_SOSPECHOSO = 2  # <= este grado (in+out) se considera mal conectado


def main():
    constructor = GraphConstructor("Comayagua,Honduras")
    constructor.construct_graph_cached("cache/comayagua_drive.graphml", network_type="drive")
    constructor.clean_graph()
    graph = constructor.graph

    pop_por_barrio, total_pop = poblacion_proyectada(escenario="central", anio_objetivo=2025)
    od_builder = ODMatrixBuilder(graph, population_by_zone=pop_por_barrio, total_population=total_pop)
    od_builder.build_od_matrix()

    nodes = od_builder.nodes
    matriz = od_builder.get_matrix()
    zonas = od_builder.get_zone_assignment_series()

    print(f"\n=== {len(nodes)} nodos-centroide de zona: grado de conectividad ===")
    sospechosos = []
    for i, n in enumerate(nodes):
        grado = graph.in_degree(n) + graph.out_degree(n)
        if grado <= GRADO_SOSPECHOSO:
            demanda_orig = matriz[i, :].sum()
            demanda_dest = matriz[:, i].sum()
            zona = zonas.loc[n] if n in zonas.index else "?"
            sospechosos.append((n, grado, zona, demanda_orig, demanda_dest))

    if not sospechosos:
        print(f"Ningún centroide tiene grado <= {GRADO_SOSPECHOSO}. Los Lirios era el único caso.")
    else:
        print(f"{len(sospechosos)} centroide(s) con grado <= {GRADO_SOSPECHOSO}:")
        for n, grado, zona, do, dd in sospechosos:
            print(f"  nodo={n}  grado={grado}  zona={zona}  "
                  f"viajes_origen={do:,.0f}  viajes_destino={dd:,.0f}")

    # Baseline + pool de candidatas real (top-100 por flujo), igual que main.py
    print("\n=== Corriendo Frank-Wolfe para el equilibrio base... ===")
    baseline = BaselineBuilder(
        graph, matriz, od_builder.nodes,
        paths=od_builder.get_paths(),
        modal_split=0.33, capacity_scale=1.0, fw_max_iter=50,
    )
    baseline.build_equilibrium()

    pool = build_candidate_pool(graph, baseline.edge_flows, top_n=100)
    pool_edges = {(ce.edge[0], ce.edge[1]) for ce in pool}

    print(f"\n=== ¿Las aristas de salida de esos centroides están en el pool top-100? ===")
    for n, grado, zona, do, dd in sospechosos:
        vecinos = list(graph.successors(n)) + list(graph.predecessors(n))
        vecinos = sorted(set(vecinos))
        for vec in vecinos:
            en_pool = (n, vec) in pool_edges or (vec, n) in pool_edges
            flujo = baseline.edge_flows.get((n, vec), baseline.edge_flows.get((vec, n), 0))
            print(f"  {n}<->{vec}: flujo={flujo:,.1f}  en_pool_top100={en_pool}")

    # Contexto: distribución de flujo para ver qué tan "outlier" es cada caso
    flujos = sorted(baseline.edge_flows.values(), reverse=True)
    print(f"\n=== Contexto: flujo del puesto #100 del pool (umbral de entrada) ===")
    print(f"  flujo mínimo para entrar al top-100: {flujos[99]:,.1f}" if len(flujos) > 99 else "N/D")
    print(f"  flujo máximo de la red: {flujos[0]:,.1f}")


if __name__ == "__main__":
    main()
