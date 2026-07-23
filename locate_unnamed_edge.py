"""
Diagnóstico puntual: ubica geográficamente la arista sin nombre
(1681023824 -> 3391839073) que aparece como crítica en el frente del
NSGA-II, y verifica si su criticidad es una arteria real o un artefacto de
la asignación de zona O-D a un nodo mal conectado.

Uso:
    python locate_unnamed_edge.py
"""

import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import osmnx as ox

from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from population_projector import poblacion_proyectada
from map_annotations import annotate_street_names

U, V = 1681023824, 3391839073


def main():
    constructor = GraphConstructor("Comayagua,Honduras")
    constructor.construct_graph_cached("cache/comayagua_drive.graphml", network_type="drive")
    constructor.clean_graph()
    graph = constructor.graph

    pop_por_barrio, total_pop = poblacion_proyectada(escenario="central", anio_objetivo=2025)
    od_builder = ODMatrixBuilder(graph, population_by_zone=pop_por_barrio, total_population=total_pop)
    od_builder.build_od_matrix()

    zonas = od_builder.get_zone_assignment_series()
    zona = zonas.loc[V]
    nodos_zona = zonas[zonas == zona].index.tolist()

    nodes_od = od_builder.nodes
    idx = nodes_od.index(V)
    matriz = od_builder.get_matrix()
    viajes_origen = matriz[idx, :].sum()
    viajes_destino = matriz[:, idx].sum()

    print(f"Arista sin nombre: {U} -> {V}")
    print(f"  osmid: {graph[U][V][0].get('osmid')}")
    print(f"  grado de {V} en el grafo: in={graph.in_degree(V)} out={graph.out_degree(V)}  "
          f"(es un extremo muerto: solo conecta con {U})")
    print(f"  zona O-D asignada a {V}: {zona}  ({len(nodos_zona)} nodos del grafo en esa zona)")
    print(f"  poblacion proyectada de la zona: {pop_por_barrio.get(zona, 'N/D')}")
    print(f"  demanda modelada en el nodo {V}: {viajes_origen:,.0f} viajes/día originados, "
          f"{viajes_destino:,.0f} viajes/día destinados")
    print(f"  -> esa demanda completa depende de esta única arista para salir a la red.")

    # Mapa zoom: la arista en rojo, el resto de nodos de la zona en naranja,
    # el resto de la red en gris tenue, con nombres de calle.
    fig, ax = ox.plot_graph(graph, node_size=0, edge_color="#bbbbbb", edge_linewidth=0.6,
                             bgcolor="white", show=False, close=False)
    annotate_street_names(ax, graph, dark=False)

    xs = [graph.nodes[n]["x"] for n in nodos_zona]
    ys = [graph.nodes[n]["y"] for n in nodos_zona]
    ax.scatter(xs, ys, s=25, color="#e69500", zorder=4,
               label=f"Nodos de {zona} ({len(nodos_zona)})")

    ex = [graph.nodes[U]["x"], graph.nodes[V]["x"]]
    ey = [graph.nodes[U]["y"], graph.nodes[V]["y"]]
    ax.plot(ex, ey, color="red", lw=3.5, zorder=5, label="Arista sin nombre (crítica)")
    ax.scatter([graph.nodes[V]["x"]], [graph.nodes[V]["y"]], s=90, color="red",
               marker="X", zorder=6, label=f"Nodo {V} (extremo muerto, centroide de zona)")

    margin = 0.006
    ax.set_xlim(min(ex) - margin, max(ex) + margin)
    ax.set_ylim(min(ey) - margin, max(ey) + margin)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Ubicación de la arista sin nombre — {zona}", fontsize=11)

    out_path = "results/comparacion_semillas/ubicacion_arista_sin_nombre.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nMapa guardado en {out_path}")


if __name__ == "__main__":
    main()
