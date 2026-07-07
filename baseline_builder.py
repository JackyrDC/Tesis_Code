"""
Construcción y visualización de la línea base de movilidad vehicular.
  1. Se estima la demanda vehicular aplicando un modal split a la demanda
     total de la matriz O-D (que ya está escalada a la población urbana).
  2. Se muestrean pares Origen-Destino proporcionales a la demanda O-D.
  3. Se pre-calculan TODOS los caminos mínimos (Dijkstra, tiempo de viaje)
     en un único pase all-pairs — luego cada viaje es un lookup O(1).
  4. Los flujos se acumulan por arista y se visualizan como mapa animado.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

_MODAL_SPLIT_DEFAULT: float = 0.30


class BaselineBuilder:
    """
    Línea base de asignación de rutas vehiculares sobre la red vial.

    Parameters
    ----------
    graph       : MultiDiGraph limpio de osmnx (debe tener 'travel_time' en aristas).
    od_matrix   : np.ndarray (n x n) — matriz O-D ya construida y escalada.
    nodes       : lista de IDs de nodo en el mismo orden que od_matrix.
    modal_split : fracción de viajes en vehículo privado (default 0.30).
    seed        : semilla para reproducibilidad del muestreo.
    paths       : dict origen -> {destino: camino} ya calculado (p. ej. desde
                  `ODMatrixBuilder.get_paths()`). Si se provee, se evita
                  recalcular el all-pairs Dijkstra por segunda vez.
    """

    def __init__(
        self,
        graph,
        od_matrix: np.ndarray,
        nodes: list,
        modal_split: float = _MODAL_SPLIT_DEFAULT,
        seed: int = 42,
        paths: dict[int, dict[int, list]] | None = None,
    ):
        self.graph = graph
        self.od_matrix = od_matrix
        self.nodes = list(nodes)
        self.n = len(self.nodes)
        self.modal_split = modal_split
        self.rng = np.random.default_rng(seed)
        self._precomputed_paths = paths

        self.n_trips = max(1, int(od_matrix.sum() * modal_split))

        self.routes: list[list] = []
        self.edge_flows: dict[tuple, float] = defaultdict(float)

        # Cache para no recalcular geometría en cada frame
        self._segments: list | None = None
        self._edge_keys: list | None = None

    # ── Muestreo de pares O-D ────────────────────────────────────────────────

    def _sample_pairs(self) -> list[tuple]:
        """Muestrea pares (orig, dest) proporcionales a la demanda O-D."""
        od = self.od_matrix.flatten().copy()
        for i in range(self.n):            # Eliminar diagonal (i == j)
            od[i * self.n + i] = 0.0
        total = od.sum()
        if total == 0:
            return []
        probs = od / total
        idx = self.rng.choice(self.n * self.n, size=self.n_trips, p=probs)
        return [(self.nodes[i // self.n], self.nodes[i % self.n]) for i in idx]

    # ── Cálculo de rutas ─────────────────────────────────────────────────────

    def build_routes(self) -> list[list]:
        """
        Pre-calcula todos los caminos mínimos (all-pairs Dijkstra) y luego
        asigna cada viaje muestreado a su ruta con lookup O(1).

        El all-pairs es O(V · E log V) total — más eficiente que E log V
        por viaje cuando se calculan miles de pares sobre el mismo grafo.
        """
        print(f"Viajes vehiculares estimados: {self.n_trips:,} "
              f"(modal split {self.modal_split:.0%})")

        if self._precomputed_paths is not None:
            print("Reutilizando caminos mínimos ya calculados (matriz O-D).")
            all_paths = self._precomputed_paths
        else:
            print(f"Pre-calculando caminos mínimos para {self.n} nodos...")
            all_paths = {}
            for k, (source, (_, paths)) in enumerate(
                nx.all_pairs_dijkstra(self.graph, weight="travel_time")
            ):
                all_paths[source] = paths
                if (k + 1) % 200 == 0 or (k + 1) == self.n:
                    print(f"  [{k + 1:>5}/{self.n}] nodos procesados...", end="\r")
            print()

        pairs = self._sample_pairs()
        failed = 0
        for orig, dest in pairs:
            path = all_paths.get(orig, {}).get(dest)
            if path and len(path) > 1:
                self.routes.append(path)
                for u, v in zip(path[:-1], path[1:]):
                    self.edge_flows[(u, v)] += 1.0
            else:
                failed += 1

        print(f"Rutas asignadas: {len(self.routes):,}  |  Sin ruta: {failed:,}")
        return self.routes

    # ── Geometría ────────────────────────────────────────────────────────────

    def _get_edge_geometry(self) -> tuple[list, list[tuple]]:
        """Devuelve (segments, edge_keys) con coordenadas (x, y) por arista."""
        if self._segments is not None:
            return self._segments, self._edge_keys

        segments = []
        edge_keys = []
        for u, v, data in self.graph.edges(data=True):
            if "geometry" in data:
                coords = list(data["geometry"].coords)
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
            else:
                xs = [self.graph.nodes[u]["x"], self.graph.nodes[v]["x"]]
                ys = [self.graph.nodes[u]["y"], self.graph.nodes[v]["y"]]
            segments.append(list(zip(xs, ys)))
            edge_keys.append((u, v))

        self._segments = segments
        self._edge_keys = edge_keys
        return segments, edge_keys

    # ── Estilo de aristas ─────────────────────────────────────────────────────

    def _edge_style(
        self,
        flows: dict[tuple, float],
        max_flow: float,
        edge_keys: list[tuple],
        cmap,
        dark: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve arrays de colores RGBA y grosores para cada arista."""
        n = len(edge_keys)
        colors = np.zeros((n, 4))
        widths = np.zeros(n)
        base_color = (0.20, 0.20, 0.20, 0.7) if dark else (0.85, 0.85, 0.85, 0.7)

        for i, (u, v) in enumerate(edge_keys):
            f = flows.get((u, v), 0.0)
            if f > 0:
                norm = min(f / max_flow, 1.0)
                colors[i] = cmap(0.15 + 0.85 * norm)
                widths[i] = 0.5 + 3.5 * norm
            else:
                colors[i] = base_color
                widths[i] = 0.4

        return colors, widths

    # ── Mapa estático ─────────────────────────────────────────────────────────

    def plot_flow_map(self) -> None:
        """Mapa estático con el flujo vehicular total acumulado por arista."""
        if not self.routes:
            print("No hay rutas calculadas. Ejecuta build_routes() primero.")
            return

        segments, edge_keys = self._get_edge_geometry()
        cmap = plt.colormaps["YlOrRd"]
        max_flow = max(self.edge_flows.values(), default=1.0)
        colors, widths = self._edge_style(self.edge_flows, max_flow, edge_keys, cmap)

        fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
        ax.set_facecolor("white")

        lc = LineCollection(segments, colors=colors, linewidths=widths, zorder=2)
        ax.add_collection(lc)

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(0, max_flow)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Flujo vehicular (viajes/día)", shrink=0.6)

        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Baseline vehicular — Comayagua\n"
            f"{len(self.routes):,} viajes asignados  |  "
            f"modal split {self.modal_split:.0%}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()

    # ── Animación ─────────────────────────────────────────────────────────────

    def animate_flow(
        self,
        n_frames: int = 60,
        interval: int = 120,
        save_path: str | None = None,
    ) -> FuncAnimation:
        """
        Anima la acumulación de flujo vehicular sobre la red vial.

        Cada frame incorpora un lote de rutas y actualiza los colores y grosores
        de las aristas. El fondo oscuro resalta el gradiente de calor.

        Parameters
        ----------
        n_frames  : número de fotogramas de la animación.
        interval  : milisegundos entre fotogramas.
        save_path : si se especifica, guarda como GIF (requiere Pillow).
                    Si es None, muestra la animación interactiva.
        """
        if not self.routes:
            print("No hay rutas calculadas. Ejecuta build_routes() primero.")
            return None

        segments, edge_keys = self._get_edge_geometry()
        cmap = plt.colormaps["YlOrRd"]
        max_flow = max(self.edge_flows.values(), default=1.0)
        total = len(self.routes)

        # ── Pre-calcular flujos acumulados por frame ─────────────────────────
        batch = max(1, total // n_frames)
        frame_flows: list[dict[tuple, float]] = []
        cum: dict[tuple, float] = defaultdict(float)

        for i, path in enumerate(self.routes):
            for u, v in zip(path[:-1], path[1:]):
                cum[(u, v)] += 1.0
            if (i + 1) % batch == 0 or i == total - 1:
                frame_flows.append(dict(cum))

        while len(frame_flows) < n_frames:
            frame_flows.append(frame_flows[-1] if frame_flows else {})
        frame_flows = frame_flows[:n_frames]

        # ── Pre-calcular colores y grosores por frame ────────────────────────
        print("Pre-calculando frames de la animación...")
        frame_styles: list[tuple[np.ndarray, np.ndarray]] = [
            self._edge_style(ff, max_flow, edge_keys, cmap, dark=True)
            for ff in frame_flows
        ]

        trips_at_frame = [
            min((f + 1) * batch, total) for f in range(n_frames)
        ]

        # ── Figura ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 10), facecolor="#111111")
        ax.set_facecolor("#111111")

        init_colors = np.full((len(segments), 4), (0.20, 0.20, 0.20, 0.7))
        lc = LineCollection(
            segments, colors=init_colors, linewidths=0.4, zorder=2
        )
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.axis("off")

        title = ax.set_title(
            "", color="white", fontsize=11, pad=10, fontweight="bold"
        )

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(0, max_flow)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Flujo (viajes/día)", shrink=0.50)
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

        plt.tight_layout()

        # ── Función de actualización ──────────────────────────────────────────

        def update(frame: int):
            colors, widths = frame_styles[frame]
            lc.set_colors(colors)
            lc.set_linewidths(widths)
            n = trips_at_frame[frame]
            title.set_text(
                f"Baseline vehicular — Comayagua  │  "
                f"{n:,} / {total:,} viajes asignados"
            )
            return lc, title

        anim = FuncAnimation(
            fig, update, frames=n_frames, interval=interval, blit=True, repeat=False
        )

        if save_path:
            print(f"Guardando animación en {save_path} ...")
            anim.save(save_path, writer="pillow", fps=max(1, 1000 // interval))
            print("Guardado.")
        else:
            plt.show()

        return anim

    # ── Resumen ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if not self.edge_flows:
            print("No hay rutas calculadas.")
            return

        flows = np.array(list(self.edge_flows.values()))
        print("\n--- Resumen Baseline Vehicular ---")
        print(f"Modal split aplicado:        {self.modal_split:.0%}")
        print(f"Viajes vehiculares totales:  {self.n_trips:,}")
        print(f"Rutas asignadas con éxito:   {len(self.routes):,}")
        print(f"Aristas con flujo > 0:       {len(self.edge_flows):,}")
        print(f"Flujo máximo por arista:     {flows.max():.0f} viajes/día")
        print(f"Flujo promedio (activas):    {flows.mean():.1f} viajes/día")
        print(f"Flujo mediano (activas):     {np.median(flows):.1f} viajes/día")

        top5 = sorted(self.edge_flows.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 aristas más cargadas:")
        for (u, v), f in top5:
            print(f"  {u} → {v}  :  {f:.0f} viajes/día")
