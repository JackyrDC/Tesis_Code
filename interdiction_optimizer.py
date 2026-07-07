

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

EdgeUV = tuple[int, int]
EdgeKey = tuple[int, int, int]

_PENALTY_MULTIPLIER = 5.0  # penalización de tiempo para pares que quedan desconectados


@dataclass
class CandidateEdge:
    """Arista candidata a interdicción, con su origen de criticidad."""

    edge: EdgeKey
    flow: float
    is_bridge: bool

    @property
    def tag(self) -> str:
        if self.is_bridge and self.flow > 0:
            return "obvia"          # mucho flujo Y sin redundancia
        if self.is_bridge:
            return "puente_bajo_flujo"   
        return "alto_flujo_redundante"   # mucho flujo pero con ruta alterna


def _is_structural_bridge(graph, edge: EdgeKey) -> bool:
    """¿Sigue existiendo un camino alterno entre los extremos de `edge` si se remueve?"""
    u, v, key = edge
    view = nx.restricted_view(graph, nodes=[], edges=[(u, v, key)])
    return not nx.has_path(view, u, v)


def build_candidate_pool(
    graph,
    edge_flows: dict[EdgeUV, float],
    top_n: int = 35,
) -> list[CandidateEdge]:
    """
    Arma el pool de candidatos: las `top_n` aristas de mayor flujo
    vehicular (según `BaselineBuilder.edge_flows`), etiquetadas además
    según si son estructuralmente únicas (puentes) o tienen redundancia.
    """
    by_flow = sorted(edge_flows.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    pool: list[CandidateEdge] = []
    for (u, v), flow in by_flow:
        # una calle puede tener aristas paralelas (u, v, key); tomamos
        # una key representativa — en una red vial limpia son raras.
        key = next(iter(graph[u][v]))
        edge = (u, v, key)
        pool.append(CandidateEdge(edge=edge, flow=flow, is_bridge=_is_structural_bridge(graph, edge)))

    return pool


def _build_edge_to_pairs(
    od_matrix: np.ndarray,
    nodes: list,
    paths: dict[int, dict[int, list]],
    pool: list[CandidateEdge],
) -> dict[EdgeUV, set[tuple[int, int]]]:
    """
    Para cada arista del pool, qué pares O-D (con demanda > 0) tienen su
    camino mínimo original pasando por ella. Se precalcula una sola vez
    para no recorrer todos los pares en cada evaluación del GA.
    """
    pool_uv = {(ce.edge[0], ce.edge[1]) for ce in pool}
    edge_to_pairs: dict[EdgeUV, set[tuple[int, int]]] = {uv: set() for uv in pool_uv}

    rows, cols = np.nonzero(od_matrix)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        o, d = nodes[i], nodes[j]
        path = paths.get(o, {}).get(d)
        if not path or len(path) < 2:
            continue
        touched = set(zip(path[:-1], path[1:])) & pool_uv
        for uv in touched:
            edge_to_pairs[uv].add((o, d))

    return edge_to_pairs


class InterdictionProblem(ElementwiseProblem):
    """Problema bi-objetivo: maximizar ΔTTV, minimizar cantidad de aristas cortadas."""

    def __init__(
        self,
        graph,
        od_matrix: np.ndarray,
        travel_times: np.ndarray,
        nodes: list,
        paths: dict[int, dict[int, list]],
        pool: list[CandidateEdge],
        budget: int = 3,
    ):
        self.graph = graph
        self.od_matrix = od_matrix
        self.travel_times = travel_times
        self.node_index = {n: i for i, n in enumerate(nodes)}
        self.pool = pool
        self.n_pool = len(pool)
        self._edge_to_pairs = _build_edge_to_pairs(od_matrix, nodes, paths, pool)

        super().__init__(n_var=budget, n_obj=2, n_constr=0, xl=0, xu=self.n_pool)

    def _decode(self, x) -> list[CandidateEdge]:
        """Vector de genes -> lista de aristas distintas a cortar (0 = sin corte)."""
        indices = {min(int(round(gene)), self.n_pool) for gene in x if int(round(gene)) > 0}
        return [self.pool[i - 1] for i in indices]

    def _evaluate(self, x, out, *args, **kwargs):
        edges_cut = self._decode(x)
        if not edges_cut:
            out["F"] = [0.0, 0.0]
            return

        cut_uv = {(ce.edge[0], ce.edge[1]) for ce in edges_cut}
        cut_uvk = [ce.edge for ce in edges_cut]

        affected_pairs: set[tuple[int, int]] = set()
        for uv in cut_uv:
            affected_pairs |= self._edge_to_pairs.get(uv, set())

        if not affected_pairs:
            out["F"] = [0.0, float(len(edges_cut))]
            return

        view = nx.restricted_view(self.graph, nodes=[], edges=cut_uvk)

        delta = 0.0
        for o, d in affected_pairs:
            i, j = self.node_index[o], self.node_index[d]
            demand = self.od_matrix[i, j]
            original_time = self.travel_times[i, j]
            try:
                new_time = nx.shortest_path_length(view, o, d, weight="travel_time")
            except nx.NetworkXNoPath:
                new_time = original_time * _PENALTY_MULTIPLIER
            delta += demand * max(0.0, new_time - original_time)

        out["F"] = [-delta, float(len(edges_cut))]


class GenerationMonitor(Callback):
    """
    Callback de pymoo: guarda, en cada generación, el frente no-dominado
    actual (`algorithm.opt`) y su Hipervolumen respecto a un punto de
    referencia FIJO — así el HV es comparable generación a generación
    (si el punto de referencia cambiara, los valores no serían comparables).
    """

    def __init__(self, ref_point: tuple[float, float]):
        super().__init__()
        self.ref_point = np.array(ref_point, dtype=float)
        self.history: list[dict] = []

    def notify(self, algorithm):
        opt = algorithm.opt
        F = opt.get("F")
        X = opt.get("X")
        hv = HV(ref_point=self.ref_point)(F) if len(F) else 0.0
        self.history.append({"gen": algorithm.n_gen, "F": F, "X": X, "hv": hv})


class LiveMapMonitor(GenerationMonitor):
    """
    Extiende GenerationMonitor para además redibujar en vivo, generación
    a generación (mientras el NSGA-II todavía está corriendo), el mapa
    vial con las aristas del frente resaltadas. Necesita los artistas de
    matplotlib ya creados (LineCollection, título) para actualizarlos in-place.
    """

    def __init__(self, ref_point, lc, title, edge_pos, base_colors, base_widths, decode_fn, cmap):
        super().__init__(ref_point)
        self.lc = lc
        self.title = title
        self.edge_pos = edge_pos
        self.base_colors = base_colors
        self.base_widths = base_widths
        self.decode_fn = decode_fn
        self.cmap = cmap

    def notify(self, algorithm):
        super().notify(algorithm)
        record = self.history[-1]

        colors = self.base_colors.copy()
        widths = self.base_widths.copy()

        counts: dict = {}
        for x in record["X"]:
            for ce in self.decode_fn(x):
                counts[ce.edge] = counts.get(ce.edge, 0) + 1

        max_count = max(counts.values(), default=1)
        for edge, count in counts.items():
            i = self.edge_pos.get(edge)
            if i is None:
                continue
            norm = count / max_count
            colors[i] = self.cmap(0.3 + 0.7 * norm)
            widths[i] = 1.5 + 3.0 * norm

        self.lc.set_colors(colors)
        self.lc.set_linewidths(widths)
        self.title.set_text(
            f"Interdicción NSGA-II — Generación {record['gen']}  │  "
            f"HV: {record['hv']:.1f}  │  frente: {len(record['X'])} soluciones"
        )

        fig = self.lc.figure
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)


class InterdictionOptimizer:
    """
    Envoltorio de alto nivel: arma el pool de candidatos, define el
    problema y corre NSGA-II para encontrar el Frente de Pareto
    daño-vs-costo de interdicción.
    """

    def __init__(
        self,
        graph,
        od_matrix: np.ndarray,
        travel_times: np.ndarray,
        nodes: list,
        paths: dict[int, dict[int, list]],
        edge_flows: dict[EdgeUV, float],
        budget: int = 3,
        top_n: int = 40,
        pop_size: int = 60,
        n_gen: int = 100,
        seed: int = 42,
        ref_point: tuple[float, float] | None = None,
        live_plot: bool = True,
    ):
        self.pool = build_candidate_pool(graph, edge_flows, top_n=top_n)
        self.problem = InterdictionProblem(
            graph, od_matrix, travel_times, nodes, paths, self.pool, budget=budget,
        )
        self.algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(vtype=float, repair=RoundingRepair()),
            mutation=PM(vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )
        self.n_gen = n_gen
        self.seed = seed
        self.live_plot = live_plot
        # peor caso posible: cero daño, un corte más de los permitidos —
        # siempre estrictamente dominado por cualquier solución real.
        self.ref_point = ref_point if ref_point is not None else (0.0, float(budget + 1))
        self.monitor = None
        self.result = None
        self.history: list[dict] = []

        self._segments: list | None = None
        self._edge_keys: list | None = None

    def _setup_live_plot(self) -> "LiveMapMonitor":
        """Arma la figura (una sola vez) que el monitor va a redibujar cada generación."""
        segments, edge_keys = self._graph_geometry()
        edge_pos = {ek: i for i, ek in enumerate(edge_keys)}
        pool_edges = {ce.edge for ce in self.pool}
        cmap = plt.colormaps["YlOrRd"]

        base_colors = np.zeros((len(segments), 4))
        base_widths = np.zeros(len(segments))
        for ek in edge_keys:
            i = edge_pos[ek]
            if ek in pool_edges:
                base_colors[i] = (0.25, 0.45, 0.75, 0.8)   # candidata, aún no cortada
                base_widths[i] = 1.2
            else:
                base_colors[i] = (0.20, 0.20, 0.20, 0.5)   # calle común
                base_widths[i] = 0.4

        plt.ion()
        fig, ax = plt.subplots(figsize=(13, 10), facecolor="#111111")
        ax.set_facecolor("#111111")
        lc = LineCollection(segments, colors=base_colors, linewidths=base_widths, zorder=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
        title = ax.set_title("Preparando NSGA-II...", color="white", fontsize=11, pad=10, fontweight="bold")
        fig.canvas.draw()
        plt.pause(0.001)

        return LiveMapMonitor(
            self.ref_point, lc, title, edge_pos, base_colors, base_widths,
            self.problem._decode, cmap,
        )

    def run(self):
        print(f"Corriendo NSGA-II: pool={len(self.pool)} aristas, "
              f"presupuesto={self.problem.n_var}, generaciones={self.n_gen}...")
        self.monitor = self._setup_live_plot() if self.live_plot else GenerationMonitor(self.ref_point)
        self.result = minimize(
            self.problem, self.algorithm, ("n_gen", self.n_gen),
            seed=self.seed, callback=self.monitor, verbose=True,
        )
        self.history = self.monitor.history
        if self.live_plot:
            plt.ioff()
        return self.result

    def hypervolume(self, ref_point: tuple[float, float] | None = None) -> float:
        """Hipervolumen del frente final — mayor es mejor (más área dominada)."""
        if self.result is None:
            raise RuntimeError("Ejecutá run() antes de calcular el hipervolumen.")
        ref = np.array(ref_point, dtype=float) if ref_point is not None else np.array(self.ref_point)
        return HV(ref_point=ref)(self.result.F)

    def edge_criticality_ranking(self) -> list[tuple[EdgeKey, int]]:
        """
        Cuenta en cuántas soluciones no-dominadas del frente aparece
        cada arista — las más repetidas son los segmentos que el
        algoritmo identifica como verdaderamente críticos.
        """
        if self.result is None:
            raise RuntimeError("Ejecutá run() antes de rankear aristas.")
        counts: dict[EdgeKey, int] = {}
        for x in self.result.X:
            for ce in self.problem._decode(x):
                counts[ce.edge] = counts.get(ce.edge, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # ── Geometría del grafo (para animar sobre el mapa) ─────────────────────

    def _graph_geometry(self) -> tuple[list, list[EdgeKey]]:
        """Devuelve (segments, edge_keys) con coordenadas (x, y) por arista, incluyendo la key."""
        if self._segments is not None:
            return self._segments, self._edge_keys

        graph = self.problem.graph
        segments, edge_keys = [], []
        for u, v, key, data in graph.edges(keys=True, data=True):
            if "geometry" in data:
                coords = list(data["geometry"].coords)
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
            else:
                xs = [graph.nodes[u]["x"], graph.nodes[v]["x"]]
                ys = [graph.nodes[u]["y"], graph.nodes[v]["y"]]
            segments.append(list(zip(xs, ys)))
            edge_keys.append((u, v, key))

        self._segments, self._edge_keys = segments, edge_keys
        return segments, edge_keys

    # ── Animación: evolución del frente sobre el mapa vial ──────────────────

    def animate_evolution(
        self,
        interval: int = 300,
        save_path: str | None = None,
    ) -> FuncAnimation | None:
        """
        Anima, generación a generación, qué aristas del pool aparecen en
        el frente de Pareto (`algorithm.opt`) sobre el mapa vial completo.
        El color/grosor de cada arista candidata refleja en cuántas
        soluciones no-dominadas de ESA generación aparece cortada.

        Requiere haber corrido run() antes (usa el historial que guarda
        GenerationMonitor).
        """
        if not self.history:
            print("No hay historial de generaciones. Ejecutá run() primero.")
            return None

        segments, edge_keys = self._graph_geometry()
        edge_pos = {ek: i for i, ek in enumerate(edge_keys)}
        pool_edges = {ce.edge for ce in self.pool}

        cmap = plt.colormaps["YlOrRd"]
        n_frames = len(self.history)

        base_colors = np.zeros((len(segments), 4))
        base_widths = np.zeros(len(segments))
        for ek in edge_keys:
            i = edge_pos[ek]
            if ek in pool_edges:
                base_colors[i] = (0.25, 0.45, 0.75, 0.8)   # candidata, no cortada esta gen.
                base_widths[i] = 1.2
            else:
                base_colors[i] = (0.20, 0.20, 0.20, 0.5)   # calle común
                base_widths[i] = 0.4

        fig, ax = plt.subplots(figsize=(13, 10), facecolor="#111111")
        ax.set_facecolor("#111111")

        lc = LineCollection(segments, colors=base_colors, linewidths=base_widths, zorder=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")

        title = ax.set_title("", color="white", fontsize=11, pad=10, fontweight="bold")

        def update(frame):
            colors = base_colors.copy()
            widths = base_widths.copy()

            record = self.history[frame]
            counts: dict[EdgeKey, int] = {}
            for x in record["X"]:
                for ce in self.problem._decode(x):
                    counts[ce.edge] = counts.get(ce.edge, 0) + 1

            max_count = max(counts.values(), default=1)
            for edge, count in counts.items():
                i = edge_pos.get(edge)
                if i is None:
                    continue
                norm = count / max_count
                colors[i] = cmap(0.3 + 0.7 * norm)
                widths[i] = 1.5 + 3.0 * norm

            lc.set_colors(colors)
            lc.set_linewidths(widths)
            title.set_text(
                f"Interdicción NSGA-II — Generación {record['gen']}/{n_frames}  │  "
                f"HV: {record['hv']:.1f}  │  frente: {len(record['X'])} soluciones"
            )
            return lc, title

        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=interval, blit=False, repeat=False,
        )

        if save_path:
            print(f"Guardando animación en {save_path} ...")
            anim.save(save_path, writer="pillow", fps=max(1, 1000 // interval))
            print("Guardado.")
        else:
            plt.show()

        return anim
