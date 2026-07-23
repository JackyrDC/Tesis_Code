"""
Mapa de presentación: aristas críticas del frente de Pareto final resaltadas
sobre la red vial completa de Comayagua, con fondo claro (para diapositivas
o el documento impreso), grosor/color por frecuencia en el frente, y las
calles principales etiquetadas directamente sobre el mapa.

Uso:
    python plot_critical_edges_map.py [run_dir]
    (por defecto usa results/run_20260721_170232)
"""

import csv
import os
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from graph_handler import GraphConstructor
from od_matrix_builder import ODMatrixBuilder
from population_projector import poblacion_proyectada

RUN_DIR_DEFAULT = "results/run_20260717_172220"
TOP_N_ETIQUETAS = 5

# Vías principales (no residenciales) que sirven de referencia de orientación
# — se etiquetan por nombre, una sola vez, sin saturar el mapa con todas las
# calles residenciales del fondo.
CLASES_VIA_PRINCIPAL = {"secondary", "secondary_link", "trunk", "trunk_link", "tertiary"}

# Landmarks puntuales que la propia tesis usa como referencia narrativa.
LANDMARKS_CATEGORIA = {"hospital"}
LANDMARKS_NOMBRE_CONTIENE = ["santa teresa"]

# Nombres legibles para aristas sin nombre en OSM, verificados a mano
# (ver Tabla 9.1 de la tesis): reemplazan el "(sin nombre: u-v)" crudo del
# pipeline. Clave = par de nodos sin importar el sentido de la arista.
NOMBRES_MANUALES = {
    frozenset({"1681023824", "3391839073"}): "Callejón del Barrio Los Lirios",
}


def cargar_frecuencias(run_dir):
    path = os.path.join(run_dir, "pareto_front_by_generation.csv")
    filas = list(csv.DictReader(open(path, encoding="utf-8")))
    ultima_gen = max(int(f["generacion"]) for f in filas)
    filas = [f for f in filas if int(f["generacion"]) == ultima_gen]
    tam_frente = len(filas)

    conteo = {}
    nombre_por_arista = {}
    for f in filas:
        aristas = [a.strip() for a in f["aristas_cortadas"].split(";") if a.strip()]
        calles = [c.strip() for c in f["calles_cortadas"].split("|") if c.strip()]
        for arista, calle in zip(aristas, calles):
            u, v, k = arista.rsplit("-", 2)
            key = (int(u), int(v), int(k))
            calle = NOMBRES_MANUALES.get(frozenset({u, v}), calle)
            conteo[key] = conteo.get(key, 0) + 1
            nombre_por_arista[key] = calle
    return conteo, nombre_por_arista, tam_frente


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR_DEFAULT

    constructor = GraphConstructor("Comayagua,Honduras")
    constructor.construct_graph_cached("cache/comayagua_drive.graphml", network_type="drive")
    constructor.clean_graph()
    graph = constructor.graph

    pop_por_barrio, total_pop = poblacion_proyectada(escenario="central", anio_objetivo=2025)
    od_builder = ODMatrixBuilder(graph, population_by_zone=pop_por_barrio, total_population=total_pop)
    pois = od_builder.get_pois()
    landmarks = [
        p for p in pois
        if p["category"] in LANDMARKS_CATEGORIA and p.get("name")
        and any(n in p["name"].lower() for n in LANDMARKS_NOMBRE_CONTIENE)
    ]

    conteo, nombre_por_arista, tam_frente = cargar_frecuencias(run_dir)
    print(f"Corrida: {run_dir}  |  {len(conteo)} aristas distintas en el frente final "
          f"({tam_frente} soluciones)")

    fig, ax = ox.plot_graph(
        graph, node_size=0, edge_color="#e5e5e5", edge_linewidth=0.4,
        bgcolor="white", show=False, close=False, figsize=(13, 11),
    )

    cmap = plt.colormaps["YlOrRd"]
    max_freq = max(conteo.values())
    norm = Normalize(vmin=0, vmax=max_freq)

    segmentos, colores, anchos = [], [], []
    for (u, v, k), freq in conteo.items():
        data = graph.get_edge_data(u, v, k)
        if data is None or "geometry" not in data:
            xs = [graph.nodes[u]["x"], graph.nodes[v]["x"]]
            ys = [graph.nodes[u]["y"], graph.nodes[v]["y"]]
            segmentos.append(list(zip(xs, ys)))
        else:
            segmentos.append(list(data["geometry"].coords))
        colores.append(cmap(norm(freq)))
        anchos.append(2.5 + 6.5 * (freq / max_freq))

    # Contorno oscuro fino para que las aristas resalten incluso las
    # delgadas (bajo freq) sobre el fondo blanco.
    ax.add_collection(LineCollection(segmentos, colors="#333333",
                                      linewidths=[a + 1.2 for a in anchos],
                                      zorder=3))
    ax.add_collection(LineCollection(segmentos, colors=colores, linewidths=anchos, zorder=4))

    # Numeritos (1, 2, 3...) sobre cada arista crítica; el detalle va en una
    # leyenda aparte, fuera del área del mapa, para no tapar la geografía.
    top = sorted(conteo.items(), key=lambda kv: -kv[1])[:TOP_N_ETIQUETAS]
    lineas_leyenda = []
    for i, ((u, v, k), freq) in enumerate(top, start=1):
        x = (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2
        y = (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2
        ax.scatter([x], [y], s=260, color="white", edgecolors="#333333",
                   linewidths=1.2, zorder=7)
        ax.annotate(str(i), xy=(x, y), ha="center", va="center", zorder=8,
                    fontsize=10, fontweight="bold", color="#333333")
        nombre = nombre_por_arista[(u, v, k)]
        pct = 100.0 * freq / tam_frente
        lineas_leyenda.append(f"{i}.  {nombre} — {pct:.0f}% del frente")

    leyenda = "\n".join(lineas_leyenda)
    ax.text(0.985, 0.015, leyenda, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right", zorder=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#333333", alpha=0.95))

    # Zoom al clúster de aristas críticas (igual criterio que el mapa en vivo
    # del optimizador: no tiene sentido encuadrar todo el municipio).
    pts = np.array([
        ((graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2,
         (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2)
        for (u, v, k) in conteo
    ])
    margin = 0.35
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = max(xmax - xmin, 0.01)
    dy = max(ymax - ymin, 0.01)
    xlim = (xmin - margin * dx, xmax + margin * dx)
    ylim = (ymin - margin * dy, ymax + margin * dy)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Vías principales dentro de la vista: se etiquetan una sola vez por
    # nombre (no cada segmento) para dar referencia sin saturar.
    ya_etiquetadas = set()
    for u, v, data in graph.edges(data=True):
        nombre = data.get("name")
        hw = data.get("highway")
        hw = hw[0] if isinstance(hw, list) else hw
        if hw not in CLASES_VIA_PRINCIPAL or not nombre:
            continue
        if isinstance(nombre, list):
            nombre = nombre[0]
        x = (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2
        y = (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2
        if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
            continue
        if nombre in ya_etiquetadas:
            continue
        ya_etiquetadas.add(nombre)
        ax.annotate(nombre, xy=(x, y), fontsize=7.5, color="#4a4a8a",
                     style="italic", zorder=5, ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Landmarks de referencia (p. ej. el hospital que cita la tesis).
    for lm in landmarks:
        if not (xlim[0] <= lm["lon"] <= xlim[1] and ylim[0] <= lm["lat"] <= ylim[1]):
            continue
        ax.scatter([lm["lon"]], [lm["lat"]], marker="P", s=180, color="#0059b3",
                   edgecolors="white", linewidths=1.0, zorder=8)
        ax.annotate(lm["name"], xy=(lm["lon"], lm["lat"]), xytext=(-10, 10),
                    textcoords="offset points", fontsize=8.5, fontweight="bold",
                    color="#0059b3", zorder=8, ha="right")

    ax.set_title("Aristas críticas del frente de Pareto — red vial de Comayagua",
                 fontsize=13, fontweight="bold", loc="left")

    # Mini-mapa de contexto: todo el municipio, con un recuadro marcando la
    # zona ampliada arriba, para orientar a quien no conoce Comayagua.
    inset = fig.add_axes([0.70, 0.55, 0.22, 0.22])
    ox.plot_graph(graph, ax=inset, node_size=0, edge_color="#cccccc",
                  edge_linewidth=0.3, bgcolor="white", show=False, close=False)
    inset.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                               fill=False, edgecolor="red", linewidth=1.5, zorder=10))
    inset.set_title("Comayagua (contexto)", fontsize=7.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#999999")

    out_path = "results/comparacion_semillas/mapa_aristas_criticas_presentacion.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Mapa guardado en {out_path}")


if __name__ == "__main__":
    main()
