"""
Análisis de fallos aleatorios (Monte Carlo) sobre la red vial.

Contrasta la interdicción DIRIGIDA (los cortes óptimos que encuentra el
NSGA-II) contra FALLOS ALEATORIOS del mismo tamaño: se remueven k aristas
al azar muchas veces y se mide el ΔTTV resultante. Permite cuantificar
cuánto más daño causa un ataque dirigido que una falla fortuita equivalente
— el contraste que pide la Fase 4 de la metodología.

La evaluación del ΔTTV reutiliza EXACTAMENTE la misma lógica que
`interdiction_optimizer._evaluate_solution` (mismo `restricted_view`, misma
fórmula demanda·Δtiempo, misma penalización por desconexión, importada de
allí), de modo que los valores dirigido y aleatorio son directamente
comparables.
"""

from __future__ import annotations

import csv

import networkx as nx
import numpy as np

from interdiction_optimizer import _PENALTY_MULTIPLIER


def _build_edge_to_pairs_all(od_matrix, nodes, paths):
    """
    Para CADA arista (u, v) de la red, qué pares O-D con demanda>0 tienen su
    camino mínimo original pasando por ella. A diferencia del equivalente del
    optimizador, no se restringe al pool de candidatas: cualquier arista puede
    fallar aleatoriamente.
    """
    edge_to_pairs: dict[tuple, set] = {}
    rows, cols = np.nonzero(od_matrix)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        o, d = nodes[i], nodes[j]
        path = paths.get(o, {}).get(d)
        if not path or len(path) < 2:
            continue
        for uv in zip(path[:-1], path[1:]):
            edge_to_pairs.setdefault(uv, set()).add((o, d))
    return edge_to_pairs


def _delta_ttv(graph, od_matrix, travel_times, node_index, edge_to_pairs, edges_cut):
    """
    ΔTTV de remover `edges_cut` (lista de (u, v, key)). Idéntico en fórmula a
    la evaluación del optimizador: solo se recalcula Dijkstra para los orígenes
    de pares afectados, y los pares que quedan desconectados se penalizan a 5×
    su tiempo original.
    """
    cut_uv = {(e[0], e[1]) for e in edges_cut}
    affected: set = set()
    for uv in cut_uv:
        affected |= edge_to_pairs.get(uv, set())
    if not affected:
        return 0.0

    by_origin: dict = {}
    for o, d in affected:
        by_origin.setdefault(o, []).append(d)

    view = nx.restricted_view(graph, nodes=[], edges=list(edges_cut))
    delta = 0.0
    for o, dests in by_origin.items():
        lengths = nx.single_source_dijkstra_path_length(view, o, weight="travel_time")
        i = node_index[o]
        for d in dests:
            j = node_index[d]
            demand = od_matrix[i, j]
            original_time = travel_times[i, j]
            new_time = lengths.get(d, original_time * _PENALTY_MULTIPLIER)
            delta += demand * max(0.0, new_time - original_time)
    return delta


def directed_optimum_by_size(result) -> dict[int, float]:
    """
    Extrae del frente de Pareto final el mejor ΔTTV para cada número de
    cortes k>0. `result` es el objeto que devuelve `InterdictionOptimizer.run()`.
    F[:,0] es -ΔTTV (pymoo minimiza) y F[:,1] el número de cortes.
    """
    directed: dict[int, float] = {}
    F = np.atleast_2d(result.F)
    for row in F:
        delta = -float(row[0])
        k = int(round(float(row[1])))
        if k > 0:
            directed[k] = max(directed.get(k, 0.0), delta)
    return directed


def run_random_failure_analysis(
    graph,
    od_matrix: np.ndarray,
    travel_times: np.ndarray,
    nodes: list,
    paths: dict,
    cut_sizes,
    n_samples: int = 500,
    seed: int = 42,
    directed_by_size: dict[int, float] | None = None,
    edge_universe: list | None = None,
    save_path: str | None = None,
) -> dict[int, dict]:
    """
    Corre el Monte Carlo de fallos aleatorios para cada tamaño de corte en
    `cut_sizes` y (si se da `directed_by_size`) lo contrasta con el óptimo
    dirigido.

    Parameters
    ----------
    cut_sizes        : tamaños de corte a evaluar (p. ej. [1, 2, 3, 4]).
    n_samples        : número de muestras aleatorias por tamaño.
    directed_by_size : {k: ΔTTV óptimo dirigido} para la columna de contraste.
    edge_universe    : aristas (u, v, key) elegibles para fallar; por defecto
                       TODAS las de la red (cualquier calle puede fallar).
    save_path        : si se da, escribe la tabla comparativa en ese CSV.

    Returns el dict {k: estadísticas}.
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    edge_to_pairs = _build_edge_to_pairs_all(od_matrix, nodes, paths)
    all_edges = edge_universe if edge_universe is not None else list(graph.edges(keys=True))
    rng = np.random.default_rng(seed)
    directed_by_size = directed_by_size or {}

    print(f"\nFallos aleatorios (Monte Carlo): {n_samples} muestras por tamaño, "
          f"universo de {len(all_edges):,} aristas.")

    results: dict[int, dict] = {}
    for k in cut_sizes:
        if k > len(all_edges):
            continue
        deltas = np.empty(n_samples)
        for s in range(n_samples):
            idx = rng.choice(len(all_edges), size=k, replace=False)
            edges_cut = [all_edges[int(t)] for t in idx]
            deltas[s] = _delta_ttv(
                graph, od_matrix, travel_times, node_index, edge_to_pairs, edges_cut
            )
        directed = directed_by_size.get(k)
        stats = {
            "num_cortes": k,
            "n_muestras": n_samples,
            "aleatorio_media": float(deltas.mean()),
            "aleatorio_std": float(deltas.std()),
            "aleatorio_max": float(deltas.max()),
            "aleatorio_p95": float(np.percentile(deltas, 95)),
            "pct_muestras_con_impacto": float((deltas > 0).mean() * 100.0),
            "dirigido": directed,
            # cuánto más daño causa el ataque dirigido que la falla promedio
            "ratio_dirigido_vs_media": (
                (directed / deltas.mean()) if directed and deltas.mean() > 0 else None
            ),
            # qué fracción de fallas aleatorias iguala o supera al dirigido
            "pct_aleatorio_supera_dirigido": (
                float((deltas >= directed).mean() * 100.0) if directed else None
            ),
        }
        results[k] = stats
        ratio = stats["ratio_dirigido_vs_media"]
        linea = (f"  k={k}: aleatorio media={stats['aleatorio_media']:,.0f} "
                 f"max={stats['aleatorio_max']:,.0f}")
        if directed is not None:
            linea += f"  |  dirigido={directed:,.0f}"
        print(linea)
        if ratio is not None:
            print(f"        -> el ataque dirigido causa {ratio:,.1f}x el dano "
                  f"de una falla aleatoria promedio")

    if save_path:
        _write_csv(results, save_path)
        print(f"  Comparación guardada en {save_path}")

    return results


def _write_csv(results: dict[int, dict], path: str) -> None:
    cols = [
        "num_cortes", "n_muestras", "aleatorio_media", "aleatorio_std",
        "aleatorio_max", "aleatorio_p95", "pct_muestras_con_impacto",
        "dirigido", "ratio_dirigido_vs_media", "pct_aleatorio_supera_dirigido",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for k in sorted(results):
            row = results[k]
            writer.writerow([
                row["num_cortes"], row["n_muestras"],
                f"{row['aleatorio_media']:.2f}", f"{row['aleatorio_std']:.2f}",
                f"{row['aleatorio_max']:.2f}", f"{row['aleatorio_p95']:.2f}",
                f"{row['pct_muestras_con_impacto']:.1f}",
                f"{row['dirigido']:.2f}" if row["dirigido"] is not None else "",
                f"{row['ratio_dirigido_vs_media']:.2f}" if row["ratio_dirigido_vs_media"] is not None else "",
                f"{row['pct_aleatorio_supera_dirigido']:.1f}" if row["pct_aleatorio_supera_dirigido"] is not None else "",
            ])
