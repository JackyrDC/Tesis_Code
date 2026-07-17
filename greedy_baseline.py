"""
Línea base greedy para contrastar con la interdicción dirigida (NSGA-II).

Dos adversarios de referencia, ambos sobre el MISMO pool de candidatas que
usa el optimizador:

  1. Greedy estático (por flujo): corta las k aristas de mayor flujo
     vehicular del equilibrio base — la estrategia "obvia" de atacar las
     calles más transitadas. Como `build_candidate_pool` ordena el pool por
     flujo descendente, equivale a tomar las primeras k candidatas.

  2. Greedy iterativo (marginal): en cada paso corta la candidata que más
     ΔTTV agrega DADO lo ya cortado. Captura efectos secuenciales, pero no
     sinergias combinatorias — el hueco que el NSGA-II debe demostrar cubrir.

Si el dirigido supera al estático, la criticidad no se reduce al volumen de
flujo (la redundancia importa); si además supera al iterativo, existen
sinergias entre cortes que la selección secuencial no captura.

La evaluación del ΔTTV reutiliza EXACTAMENTE la de `random_failures`
(`_delta_ttv`, que a su vez replica la fórmula del optimizador), de modo que
dirigido, aleatorio y greedy son directamente comparables.
"""

from __future__ import annotations

import csv

from random_failures import _build_edge_to_pairs_all, _delta_ttv
from results_exporter import _edge_street_name


def _greedy_static(pool, cut_sizes, evaluate):
    """
    ΔTTV de cortar las k candidatas de mayor flujo, para cada k en
    `cut_sizes`. Devuelve {k: (delta, [aristas cortadas])}.
    """
    ranked = sorted(pool, key=lambda c: c.flow, reverse=True)
    results = {}
    for k in cut_sizes:
        if k > len(ranked):
            continue
        edges = [c.edge for c in ranked[:k]]
        results[k] = (evaluate(edges), edges)
    return results


def _greedy_iterative(pool, max_k, evaluate):
    """
    Selección secuencial: en cada paso se prueba agregar cada candidata
    restante al conjunto ya cortado y se fija la que maximiza el ΔTTV
    acumulado. Devuelve {k: (delta acumulado, [aristas cortadas])} para
    k = 1..max_k.
    """
    remaining = list(pool)
    chosen: list = []
    results = {}
    for step in range(1, min(max_k, len(pool)) + 1):
        best_delta, best_cand = -1.0, None
        for cand in remaining:
            delta = evaluate([c.edge for c in chosen] + [cand.edge])
            if delta > best_delta:
                best_delta, best_cand = delta, cand
        chosen.append(best_cand)
        remaining.remove(best_cand)
        results[step] = (best_delta, [c.edge for c in chosen])
    return results


def run_greedy_baseline_analysis(
    graph,
    od_matrix,
    travel_times,
    nodes: list,
    paths: dict,
    pool: list,
    cut_sizes,
    directed_by_size: dict[int, float] | None = None,
    save_path: str | None = None,
) -> dict[int, dict]:
    """
    Corre ambos greedy sobre el pool de candidatas y (si se da
    `directed_by_size`) los contrasta con el óptimo dirigido del NSGA-II.

    Parameters
    ----------
    pool             : lista de `CandidateEdge` (la misma de `optimizer.pool`).
    cut_sizes        : tamaños de corte a evaluar (p. ej. [1, 2, 3, 4, 5]).
    directed_by_size : {k: ΔTTV óptimo dirigido} para la columna de contraste
                       (el dict que devuelve `directed_optimum_by_size`).
    save_path        : si se da, escribe la tabla comparativa en ese CSV.

    Returns el dict {k: estadísticas}.
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    edge_to_pairs = _build_edge_to_pairs_all(od_matrix, nodes, paths)
    directed_by_size = directed_by_size or {}
    cut_sizes = sorted(k for k in cut_sizes if k > 0)
    if not cut_sizes:
        return {}

    def evaluate(edges_cut):
        return _delta_ttv(
            graph, od_matrix, travel_times, node_index, edge_to_pairs, edges_cut
        )

    n_evals_iter = sum(len(pool) - s for s in range(min(max(cut_sizes), len(pool))))
    print(f"\nLinea base greedy: pool de {len(pool)} candidatas, "
          f"~{n_evals_iter:,} evaluaciones para el iterativo.")

    static = _greedy_static(pool, cut_sizes, evaluate)
    iterative = _greedy_iterative(pool, max(cut_sizes), evaluate)

    results: dict[int, dict] = {}
    for k in cut_sizes:
        if k not in static or k not in iterative:
            continue
        static_delta, static_edges = static[k]
        iter_delta, iter_edges = iterative[k]
        directed = directed_by_size.get(k)
        stats = {
            "num_cortes": k,
            "greedy_flujo": static_delta,
            "greedy_iterativo": iter_delta,
            "dirigido": directed,
            # >1 significa que el NSGA-II supera a la estrategia greedy
            "ratio_dirigido_vs_greedy_flujo": (
                (directed / static_delta) if directed and static_delta > 0 else None
            ),
            "ratio_dirigido_vs_greedy_iter": (
                (directed / iter_delta) if directed and iter_delta > 0 else None
            ),
            "calles_greedy_flujo": " | ".join(
                _edge_street_name(graph, e) for e in static_edges
            ),
            "calles_greedy_iterativo": " | ".join(
                _edge_street_name(graph, e) for e in iter_edges
            ),
        }
        results[k] = stats

        linea = (f"  k={k}: greedy flujo={static_delta:,.0f}  "
                 f"greedy iterativo={iter_delta:,.0f}")
        if directed is not None:
            linea += f"  |  dirigido={directed:,.0f}"
        print(linea)
        ratio = stats["ratio_dirigido_vs_greedy_iter"]
        if ratio is not None:
            print(f"        -> el dirigido logra {ratio:,.2f}x el dano "
                  f"del mejor greedy iterativo")

    if save_path:
        _write_csv(results, save_path)
        print(f"  Comparacion guardada en {save_path}")

    return results


def _write_csv(results: dict[int, dict], path: str) -> None:
    cols = [
        "num_cortes", "greedy_flujo", "greedy_iterativo", "dirigido",
        "ratio_dirigido_vs_greedy_flujo", "ratio_dirigido_vs_greedy_iter",
        "calles_greedy_flujo", "calles_greedy_iterativo",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for k in sorted(results):
            row = results[k]
            writer.writerow([
                row["num_cortes"],
                f"{row['greedy_flujo']:.2f}",
                f"{row['greedy_iterativo']:.2f}",
                f"{row['dirigido']:.2f}" if row["dirigido"] is not None else "",
                f"{row['ratio_dirigido_vs_greedy_flujo']:.2f}"
                if row["ratio_dirigido_vs_greedy_flujo"] is not None else "",
                f"{row['ratio_dirigido_vs_greedy_iter']:.2f}"
                if row["ratio_dirigido_vs_greedy_iter"] is not None else "",
                row["calles_greedy_flujo"],
                row["calles_greedy_iterativo"],
            ])
