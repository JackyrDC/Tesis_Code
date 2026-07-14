"""
Persistencia de resultados de una corrida de interdicción NSGA-II.

Cada corrida produce una carpeta `results/run_<fecha>/` con:
  - metadata.json                     : parámetros + resumen de resultados
  - pareto_front_by_generation.csv    : el frente no-dominado de CADA generación,
                                        con las aristas cortadas por nombre de calle
  - mapa_frente_final.png             : red vial con las aristas críticas resaltadas
  - convergencia_hv.png               : hipervolumen por generación (convergencia)
  - frente_pareto.png                 : ΔTTV vs. número de cortes (frente final)

El objetivo es reproducibilidad: cada figura de la tesis queda rastreable a
una corrida identificada por fecha, semilla y parámetros.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def _edge_street_name(graph, edge) -> str:
    """
    Nombre de calle legible de una arista (u, v, key). `name` en OSM puede
    ser str, lista (vía con varios nombres) o faltar; sin nombre se cae al
    par de nodos para no perder la referencia.
    """
    u, v = edge[0], edge[1]
    key = edge[2] if len(edge) > 2 else None
    data = graph.get_edge_data(u, v, key) if key is not None else graph.get_edge_data(u, v)
    name = data.get("name") if data else None
    if isinstance(name, (list, tuple)):
        name = ", ".join(str(n) for n in name if n)
    if isinstance(name, str) and name.strip():
        return name
    return f"(sin nombre: {u}-{v})"


def _solutions(record) -> list[tuple]:
    """
    Normaliza un registro de generación (`{gen, F, X, hv}`) a una lista de
    (delta_ttv, n_cuts, X_row). F[:,0] es -ΔTTV (pymoo minimiza), F[:,1] el
    número de cortes.
    """
    F = np.atleast_2d(record["F"]) if len(record["F"]) else np.empty((0, 2))
    X = np.atleast_2d(record["X"]) if len(record["X"]) else np.empty((0, 0))
    out = []
    for i in range(len(F)):
        out.append((float(-F[i, 0]), float(F[i, 1]), X[i]))
    return out


def _write_pareto_csv(optimizer, path: str) -> None:
    """CSV con el frente no-dominado de cada generación."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generacion", "hipervolumen", "id_solucion",
            "delta_ttv", "num_cortes", "calles_cortadas", "aristas_cortadas",
        ])
        for record in optimizer.history:
            gen = record["gen"]
            hv = record["hv"]
            for sol_id, (delta, n_cuts, x) in enumerate(_solutions(record)):
                cut_edges = [ce.edge for ce in optimizer.problem._decode(x)]
                streets = " | ".join(
                    _edge_street_name(optimizer.problem.ctx.graph, e) for e in cut_edges
                )
                edge_ids = "; ".join(
                    f"{e[0]}-{e[1]}-{e[2] if len(e) > 2 else 0}" for e in cut_edges
                )
                writer.writerow([
                    gen, f"{hv:.4f}", sol_id,
                    f"{delta:.2f}", int(n_cuts), streets, edge_ids,
                ])


def _save_convergence_plot(optimizer, path: str) -> None:
    """Hipervolumen por generación: evidencia de convergencia del NSGA-II."""
    gens = [r["gen"] for r in optimizer.history]
    hvs = [r["hv"] for r in optimizer.history]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.plot(gens, hvs, marker="o", ms=3, color="#cc5500", lw=1.5)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Hipervolumen")
    ax.set_title("Convergencia del NSGA-II (hipervolumen del frente)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_pareto_scatter(optimizer, path: str) -> None:
    """Frente de Pareto final: ΔTTV vs. número de cortes."""
    final = _solutions(optimizer.history[-1])
    deltas = [s[0] for s in final]
    cuts = [s[1] for s in final]
    order = np.argsort(cuts)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.scatter(cuts, deltas, s=45, color="#1f4e79", zorder=3)
    ax.plot(np.array(cuts)[order], np.array(deltas)[order],
            color="#1f4e79", lw=1.0, alpha=0.5, zorder=2)
    ax.set_xlabel("Número de aristas cortadas")
    ax.set_ylabel("ΔTTV (daño: veh·s adicionales/día)")
    ax.set_title("Frente de Pareto final — daño vs. costo de interdicción")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_front_map(optimizer, path: str) -> None:
    """Mapa vial con las aristas del frente final resaltadas."""
    renderer = optimizer._base_map_figure()
    renderer.draw(optimizer.history[-1])
    fig = renderer.lc.figure
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def _build_metadata(optimizer, params: dict | None, run_id: str, timestamp: str) -> dict:
    final = _solutions(optimizer.history[-1])
    graph = optimizer.problem.ctx.graph
    ranking = optimizer.edge_criticality_ranking()[:10]
    n_gen = len(optimizer.history)
    elapsed = getattr(optimizer, "elapsed_seconds", None)
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "parametros": params or {},
        "resultados": {
            "tiempo_ejecucion_seg": round(elapsed, 3) if elapsed is not None else None,
            "tiempo_promedio_por_generacion_seg": (
                round(elapsed / n_gen, 3) if elapsed is not None and n_gen else None
            ),
            "hipervolumen_final": round(float(optimizer.hypervolume()), 4),
            "num_generaciones": n_gen,
            "tamano_frente_final": len(final),
            "tamano_pool_candidatas": len(optimizer.pool),
            "aristas_criticas_top": [
                {
                    "arista": f"{e[0]}-{e[1]}-{e[2] if len(e) > 2 else 0}",
                    "calle": _edge_street_name(graph, e),
                    "frecuencia_en_frente": count,
                }
                for e, count in ranking
            ],
        },
    }


def export_run(optimizer, base_dir: str = "results", params: dict | None = None,
               run_id: str | None = None, timestamp: str | None = None) -> str:
    """
    Vuelca todos los artefactos de la corrida a `base_dir/run_<id>/` y
    devuelve la ruta de esa carpeta.

    Parameters
    ----------
    optimizer : InterdictionOptimizer ya ejecutado (con history y result).
    params    : dict de parámetros de la corrida (para metadata.json).
    run_id    : identificador de la carpeta; por defecto la fecha-hora local.
    timestamp : marca de tiempo ISO; por defecto la actual.
    """
    if not optimizer.history:
        print("No hay historial de generaciones. Ejecutá run() antes de exportar.")
        return ""

    now = datetime.now()
    run_id = run_id or now.strftime("%Y%m%d_%H%M%S")
    timestamp = timestamp or now.isoformat(timespec="seconds")

    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "pareto_front_by_generation.csv")
    _write_pareto_csv(optimizer, csv_path)
    _save_convergence_plot(optimizer, os.path.join(run_dir, "convergencia_hv.png"))
    _save_pareto_scatter(optimizer, os.path.join(run_dir, "frente_pareto.png"))
    _save_front_map(optimizer, os.path.join(run_dir, "mapa_frente_final.png"))

    metadata = _build_metadata(optimizer, params, run_id, timestamp)
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Resultados guardados en {run_dir}")
    return run_dir
