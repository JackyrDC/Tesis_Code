

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from scipy.cluster.hierarchy import fclusterdata

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from map_annotations import annotate_pois, annotate_street_names

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


# ── Evaluación de soluciones (compartida por proceso principal y workers) ────

@dataclass
class _EvalContext:
    """Datos mínimos que necesita un proceso para evaluar una solución."""

    graph: object
    od_matrix: np.ndarray
    travel_times: np.ndarray
    node_index: dict
    pool_edges: list[EdgeKey]
    edge_to_pairs: dict[EdgeUV, set[tuple[int, int]]]


def _decode_indices(x, n_pool: int) -> list[int]:
    """Vector de genes -> índices 1-based distintos del pool (0 = sin corte)."""
    return sorted({min(int(round(g)), n_pool) for g in x if int(round(g)) > 0})


def _evaluate_solution(ctx: _EvalContext, x) -> tuple[float, float]:
    """
    Evalúa una solución: (-ΔTTV, #cortes).

    Los pares afectados se agrupan por ORIGEN y se corre un único Dijkstra
    single-source por origen afectado, en lugar de un Dijkstra por par O-D:
    con Z zonas el costo por evaluación queda acotado por Z Dijkstras.
    """
    idxs = _decode_indices(x, len(ctx.pool_edges))
    if not idxs:
        return 0.0, 0.0

    edges_cut = [ctx.pool_edges[i - 1] for i in idxs]
    cut_uv = {(u, v) for u, v, _ in edges_cut}

    affected: set[tuple[int, int]] = set()
    for uv in cut_uv:
        affected |= ctx.edge_to_pairs.get(uv, set())
    if not affected:
        return 0.0, float(len(edges_cut))

    by_origin: dict[int, list[int]] = {}
    for o, d in affected:
        by_origin.setdefault(o, []).append(d)

    view = nx.restricted_view(ctx.graph, nodes=[], edges=edges_cut)
    delta = 0.0
    for o, dests in by_origin.items():
        lengths = nx.single_source_dijkstra_path_length(view, o, weight="travel_time")
        i = ctx.node_index[o]
        for d in dests:
            j = ctx.node_index[d]
            demand = ctx.od_matrix[i, j]
            original_time = ctx.travel_times[i, j]
            new_time = lengths.get(d, original_time * _PENALTY_MULTIPLIER)
            delta += demand * max(0.0, new_time - original_time)

    return -delta, float(len(edges_cut))


# El contexto (incluido el grafo) se serializa UNA sola vez por worker, vía
# el initializer del Pool, y queda como global del proceso hijo — así no se
# re-transfiere el grafo en cada generación ni en cada evaluación.
_WORKER_CTX: _EvalContext | None = None


def _worker_init(ctx: _EvalContext) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _worker_evaluate(x) -> tuple[float, float]:
    return _evaluate_solution(_WORKER_CTX, x)


class InterdictionProblem(Problem):
    """
    Problema bi-objetivo (evaluación por lotes): maximizar ΔTTV, minimizar
    la cantidad de aristas cortadas. Si el optimizador le inyecta un
    `map` paralelo (ver `set_parallel_map`), la población de cada
    generación se evalúa repartida entre procesos.
    """

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
        self.pool = pool
        self.n_pool = len(pool)
        self.ctx = _EvalContext(
            graph=graph,
            od_matrix=od_matrix,
            travel_times=travel_times,
            node_index={n: i for i, n in enumerate(nodes)},
            pool_edges=[ce.edge for ce in pool],
            edge_to_pairs=_build_edge_to_pairs(od_matrix, nodes, paths, pool),
        )
        self._parallel_map = None

        super().__init__(n_var=budget, n_obj=2, n_ieq_constr=0, xl=0, xu=self.n_pool)

    def set_parallel_map(self, map_fn) -> None:
        """Inyecta (o retira, con None) el `Pool.map` a usar en _evaluate."""
        self._parallel_map = map_fn

    def _decode(self, x) -> list[CandidateEdge]:
        """Vector de genes -> lista de aristas distintas a cortar (0 = sin corte)."""
        return [self.pool[i - 1] for i in _decode_indices(x, self.n_pool)]

    def _evaluate(self, X, out, *args, **kwargs):
        rows = list(X)
        if self._parallel_map is not None:
            F = self._parallel_map(_worker_evaluate, rows)
        else:
            F = [_evaluate_solution(self.ctx, x) for x in rows]
        out["F"] = np.array(F, dtype=float)


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


class QueueMonitor(GenerationMonitor):
    """
    Callback del lado del NSGA-II cuando este corre en un hilo aparte
    (mapa en vivo): además de guardar el historial, publica cada registro
    de generación en una cola para que el hilo principal — único dueño de
    la GUI, matplotlib no es thread-safe — lo consuma y redibuje.
    """

    def __init__(self, ref_point, records: queue.Queue):
        super().__init__(ref_point)
        self.records = records

    def notify(self, algorithm):
        super().notify(algorithm)
        self.records.put(self.history[-1])


class FrontRenderer:
    """
    Resalta sobre el mapa vial las aristas del frente de una generación.
    Debe usarse SOLO desde el hilo principal. Es el dibujante compartido
    del mapa en vivo, del explorador post-ejecución (`explore_history`)
    y del replay (`animate_evolution`).
    """

    def __init__(self, lc, title, edge_pos, base_colors, base_widths, decode_fn, cmap):
        self.lc = lc
        self.title = title
        self.edge_pos = edge_pos
        self.base_colors = base_colors
        self.base_widths = base_widths
        self.decode_fn = decode_fn
        self.cmap = cmap

    def draw(self, record: dict) -> None:
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
        self.lc.figure.canvas.draw_idle()


class InterdictionOptimizer:
    """
    Envoltorio de alto nivel: arma el pool de candidatos, define el
    problema y corre NSGA-II para encontrar el Frente de Pareto
    daño-vs-costo de interdicción.

    Con `n_jobs > 1` la evaluación de cada generación se reparte entre
    procesos (multiprocessing.Pool). IMPORTANTE en Windows: el script que
    llama a run() debe estar protegido con `if __name__ == "__main__":`,
    porque cada worker re-importa el módulo principal al arrancar.
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
        live_update_every: int = 5,
        n_jobs: int | None = None,
        pois: list[dict] | None = None,
    ):
        # POIs {lat, lon, category, name} para anotar los mapas (opcional).
        self.pois = pois or []
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
        self.live_update_every = live_update_every
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        # peor caso posible: cero daño, un corte más de los permitidos —
        # siempre estrictamente dominado por cualquier solución real.
        self.ref_point = ref_point if ref_point is not None else (0.0, float(budget + 1))
        self.monitor = None
        self.result = None
        self.history: list[dict] = []
        self.elapsed_seconds: float | None = None

        self._segments: list | None = None
        self._edge_keys: list | None = None

    def _zoom_to_candidates(self, ax, t: float = 0.008, margin: float = 0.30) -> None:
        """
        Acerca la vista al mayor clúster de aristas candidatas — sin esto,
        el autoscale encuadra el municipio completo y la zona de interdicción
        queda como un punto en el centro del mapa.

        Cada candidata se representa por el punto medio de sus extremos y se
        agrupa por single-linkage con umbral `t` (en grados: 0.008° ≈ 880 m a
        la latitud de Comayagua), que equivale a componentes conexas dentro de
        un radio — así una arteria periférica de alto flujo no arruina el
        encuadre. `margin` es el margen proporcional alrededor del clúster.
        """
        if not self.pool:
            return

        graph = self.problem.ctx.graph
        mids = np.array([
            ((graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2.0,
             (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2.0)
            for u, v, _ in (ce.edge for ce in self.pool)
        ])

        if len(mids) < 2:
            pts = mids
        else:
            labels = fclusterdata(mids, t=t, criterion="distance", method="single")
            vals, counts = np.unique(labels, return_counts=True)
            pts = mids[labels == vals[counts.argmax()]]

        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        # bbox degenerado si el clúster es colineal (una sola calle recta)
        dx = max(xmax - xmin, 1e-4)
        dy = max(ymax - ymin, 1e-4)
        ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
        ax.set_ylim(ymin - margin * dy, ymax + margin * dy)

    def _base_map_figure(self) -> FrontRenderer:
        """
        Construye la figura base (red vial + candidatas en azul + anotaciones
        + zoom al clúster de interdicción) y devuelve el FrontRenderer que
        sabe resaltar cualquier frente sobre ella. Compartida por el mapa en
        vivo, `explore_history` y `animate_evolution`.
        """
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

        fig, ax = plt.subplots(figsize=(13, 10), facecolor="#111111")
        ax.set_facecolor("#111111")
        lc = LineCollection(segments, colors=base_colors, linewidths=base_widths, zorder=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
        self._zoom_to_candidates(ax)
        annotate_street_names(ax, self.problem.ctx.graph, dark=True)
        annotate_pois(ax, self.pois, dark=True)
        title = ax.set_title("", color="white", fontsize=11, pad=10, fontweight="bold")

        return FrontRenderer(
            lc, title, edge_pos, base_colors, base_widths, self.problem._decode, cmap,
        )

    def run(self):
        print(f"Corriendo NSGA-II: pool={len(self.pool)} aristas, "
              f"presupuesto={self.problem.n_var}, generaciones={self.n_gen}, "
              f"procesos={self.n_jobs}...")
        records: queue.Queue = queue.Queue()
        self.monitor = (QueueMonitor(self.ref_point, records) if self.live_plot
                        else GenerationMonitor(self.ref_point))

        mp_pool = None
        if self.n_jobs > 1:
            mp_pool = mp.Pool(self.n_jobs, initializer=_worker_init, initargs=(self.problem.ctx,))
            self.problem.set_parallel_map(mp_pool.map)
        start = time.perf_counter()
        try:
            if self.live_plot:
                self.result = self._run_with_live_map(records)
            else:
                self.result = self._minimize()
        finally:
            if mp_pool is not None:
                mp_pool.close()
                mp_pool.join()
            self.problem.set_parallel_map(None)
        # Tiempo de pared de la optimización (incluye setup/teardown del Pool).
        self.elapsed_seconds = time.perf_counter() - start

        self.history = self.monitor.history
        return self.result

    def _minimize(self):
        return minimize(
            self.problem, self.algorithm, ("n_gen", self.n_gen),
            seed=self.seed, callback=self.monitor, verbose=True,
        )

    def _run_with_live_map(self, records: queue.Queue):
        """
        Corre el NSGA-II en un hilo aparte mientras el hilo principal bombea
        el event loop de la GUI: el mapa queda navegable (pan/zoom) durante
        toda la ejecución, no solo en los instantes de redibujado. El pan y
        el zoom del usuario persisten entre generaciones porque solo se
        actualizan colores/grosores, nunca los límites de los ejes.

        El hilo del GA pasa la mayor parte del tiempo esperando al Pool de
        procesos, así que no compite por el GIL con el dibujado.
        """
        plt.ion()
        renderer = self._base_map_figure()
        renderer.title.set_text("Preparando NSGA-II...")
        fig = renderer.lc.figure
        fig.canvas.draw()
        plt.pause(0.001)

        outcome: dict = {}

        def _optimize():
            try:
                outcome["result"] = self._minimize()
            except BaseException as exc:   # re-lanzada en el hilo principal
                outcome["error"] = exc

        worker = threading.Thread(target=_optimize, name="nsga2", daemon=True)
        worker.start()

        latest: dict | None = None
        drawn_gen = 0
        while worker.is_alive() or not records.empty():
            try:
                while True:                # quedarse solo con el más reciente
                    latest = records.get_nowait()
            except queue.Empty:
                pass

            if not plt.fignum_exists(fig.number):
                # Ventana cerrada por el usuario: el GA sigue hasta terminar.
                worker.join(timeout=0.2)
                continue

            if latest is not None and latest["gen"] > drawn_gen and (
                latest["gen"] == 1 or latest["gen"] % self.live_update_every == 0
            ):
                renderer.draw(latest)
                drawn_gen = latest["gen"]
            plt.pause(0.05)                # bombea eventos de la GUI

        worker.join()
        if "error" in outcome:
            raise outcome["error"]

        # Garantizar que quede dibujada la generación final.
        if latest is not None and plt.fignum_exists(fig.number):
            renderer.draw(latest)
            plt.pause(0.001)
        plt.ioff()
        return outcome["result"]

    def explore_history(self):
        """
        Explorador interactivo POST-ejecución: el mismo mapa del frente con
        un slider para navegar generación a generación, con pan/zoom libres
        (el algoritmo ya terminó, la GUI no compite con nadie). Bloquea
        hasta que se cierre la ventana. Requiere haber corrido run().
        """
        if not self.history:
            print("No hay historial de generaciones. Ejecutá run() primero.")
            return None

        renderer = self._base_map_figure()
        fig = renderer.lc.figure
        fig.subplots_adjust(bottom=0.09)

        sax = fig.add_axes([0.18, 0.035, 0.64, 0.03])
        sax.set_facecolor("#333333")
        slider = Slider(
            sax, "Generación", 1, len(self.history),
            valinit=len(self.history), valstep=1, color="#cc5500",
        )
        slider.label.set_color("white")
        slider.valtext.set_color("white")
        slider.on_changed(lambda val: renderer.draw(self.history[int(val) - 1]))

        renderer.draw(self.history[-1])
        plt.show()
        return slider

    def hypervolume(self, ref_point: tuple[float, float] | None = None) -> float:
        """Hipervolumen del frente final — mayor es mejor (más área dominada)."""
        if self.result is None:
            raise RuntimeError("Ejecutá run() antes de calcular el hipervolumen.")
        ref = np.array(ref_point, dtype=float) if ref_point is not None else np.array(self.ref_point)
        return HV(ref_point=ref)(self.result.F)

    def save_run(self, base_dir: str = "results", params: dict | None = None) -> str:
        """
        Persiste la corrida (CSV del frente por generación, metadata y mapas)
        en `base_dir/run_<fecha>/`. Devuelve la ruta de la carpeta creada.
        """
        from results_exporter import export_run
        return export_run(self, base_dir=base_dir, params=params)

    def edge_criticality_ranking(self) -> list[tuple[EdgeKey, int]]:
        """
        Cuenta en cuántas soluciones no-dominadas del frente aparece
        cada arista — las más repetidas son los segmentos que el
        algoritmo identifica como verdaderamente críticos.
        """
        if self.result is None:
            raise RuntimeError("Ejecutá run() antes de rankear aristas.")
        counts: dict[EdgeKey, int] = {}
        for x in np.atleast_2d(self.result.X):
            for ce in self.problem._decode(x):
                counts[ce.edge] = counts.get(ce.edge, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # ── Geometría del grafo (para animar sobre el mapa) ─────────────────────

    def _graph_geometry(self) -> tuple[list, list[EdgeKey]]:
        """Devuelve (segments, edge_keys) con coordenadas (x, y) por arista, incluyendo la key."""
        if self._segments is not None:
            return self._segments, self._edge_keys

        graph = self.problem.ctx.graph
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

        renderer = self._base_map_figure()
        fig = renderer.lc.figure
        n_frames = len(self.history)

        def update(frame):
            renderer.draw(self.history[frame])
            return renderer.lc, renderer.title

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
