"""
Asignación de tráfico por equilibrio de usuario (Wardrop) mediante el
algoritmo de Frank-Wolfe.

Reemplaza la asignación "todo-o-nada" (cada viaje por su ruta más corta a
velocidad libre) por un equilibrio en el que el costo de cada arista CRECE
con el flujo (función BPR). En equilibrio ningún conductor puede reducir su
tiempo cambiando de ruta — es la respuesta del "defensor" en el juego de
interdicción.

Referencias
-----------
- Wardrop, J.G. (1952). Some theoretical aspects of road traffic research.
  Proc. Institution of Civil Engineers, Part II, 1(3), 325-362.
  DOI: 10.1680/ipeds.1952.11259
- LeBlanc, Morlok, Pierskalla (1975). An efficient approach to solving the
  road network equilibrium traffic assignment problem. Transportation
  Research, 9(5), 309-318. DOI: 10.1016/0041-1647(75)90030-1
- Bureau of Public Roads (1964). Traffic Assignment Manual (función BPR,
  alpha=0.15, beta=4).
- Transportation Research Board (2016). Highway Capacity Manual, 6th ed.
  DOI: 10.17226/24798 (flujo de saturación base ~1900 veh/h/carril).

Modelo ESTÁTICO (sin franjas horarias): se asigna la demanda O-D diaria ya
ajustada por modal split en un único equilibrio. El cociente v/c resultante
es un ÍNDICE DE CONGESTIÓN RELATIVA / saturación estructural, no una
saturación horaria literal.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

# ── Parámetros BPR (Bureau of Public Roads, 1964) ────────────────────────────
BPR_ALPHA: float = 0.15
BPR_BETA: float = 4.0

# ── Capacidad efectiva por clase de vía (veh/h, por sentido) ─────────────────
# Supuesto de modelado: HCM (flujo de saturación ~1900 veh/h/carril) escalado
# por carriles típicos por clase y reducido por interrupciones urbanas.
CAPACITY_BY_HIGHWAY: dict[str, float] = {
    "motorway": 2200.0,
    "trunk": 2000.0,
    "primary": 1800.0,
    "secondary": 1200.0,
    "tertiary": 800.0,
    "residential": 600.0,
    "unclassified": 600.0,
    "living_street": 300.0,
}
DEFAULT_CAPACITY: float = 600.0  # clase desconocida/faltante -> local conservador

def highway_capacity(hw) -> float:
    """
    Capacidad (veh/h) de una arista según su tag OSM `highway`.

    `highway` puede venir como lista (p. ej. ['residential', 'tertiary']);
    en ese caso se toma la clase de MAYOR capacidad. Los sufijos '_link'
    (rampas) se mapean a su clase base.
    """
    if isinstance(hw, (list, tuple)):
        return max((highway_capacity(h) for h in hw), default=DEFAULT_CAPACITY)
    if not isinstance(hw, str):
        return DEFAULT_CAPACITY
    hw = hw[:-5] if hw.endswith("_link") else hw
    return CAPACITY_BY_HIGHWAY.get(hw, DEFAULT_CAPACITY)


class FrankWolfeAssignment:
    """
    Equilibrio de usuario (Wardrop) por Frank-Wolfe sobre un grafo vial.

    Parameters
    ----------
    graph          : MultiDiGraph de osmnx con 'travel_time' (free-flow) y
                     'highway' en las aristas.
    od_matrix      : np.ndarray (Z x Z) — demanda ya escalada (p. ej. la
                     demanda vehicular tras aplicar el modal split).
    nodes          : lista de IDs de nodo (centroides de zona) en el mismo
                     orden que las filas/columnas de `od_matrix`.
    alpha, beta    : parámetros de la función BPR.
    capacity_scale : factor global de calibración de la capacidad. Sube o baja
                     todas las capacidades a la vez para llevar el v/c mediano
                     a un rango donde la BPR "muerda" (~0.8).
    max_iter       : máximo de iteraciones de Frank-Wolfe.
    tol            : tolerancia del relative gap para declarar convergencia.
    verbose        : imprime el gap por iteración.
    """

    def __init__(
        self,
        graph,
        od_matrix: np.ndarray,
        nodes: list,
        alpha: float = BPR_ALPHA,
        beta: float = BPR_BETA,
        capacity_scale: float = 1.0,
        max_iter: int = 30,
        tol: float = 1e-4,
        verbose: bool = True,
    ):
        self.graph = graph
        self.od = np.asarray(od_matrix, dtype=float)
        self.nodes = list(nodes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.capacity_scale = float(capacity_scale)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = verbose

        # ── Enumeración de aristas y atributos vectorizados ──────────────────
        self.edge_list: list[tuple] = list(graph.edges(keys=True))
        m = len(self.edge_list)
        self.t0 = np.empty(m, dtype=float)     # tiempo free-flow por arista
        self.cap = np.empty(m, dtype=float)    # capacidad (veh/h) por arista

        # Para cargar flujo sobre una ruta hay que elegir, entre aristas
        # paralelas (u, v, key), la de menor costo — la misma que usaría
        # Dijkstra. Se indexan las keys por par (u, v).
        self.uv_keys: dict[tuple, list[tuple[object, int]]] = {}

        for i, (u, v, k) in enumerate(self.edge_list):
            data = graph[u][v][k]
            t0 = data.get("travel_time")
            if t0 is None:  # fallback: longitud [m] / velocidad [m/s]
                speed = data.get("speed_kph", 30.0) / 3.6
                t0 = data.get("length", 0.0) / max(speed, 1e-6)
            self.t0[i] = t0
            self.cap[i] = highway_capacity(data.get("highway")) * self.capacity_scale
            self.uv_keys.setdefault((u, v), []).append((k, i))

        self.cap = np.maximum(self.cap, 1e-6)  # evita división por cero
        self.flow = np.zeros(m, dtype=float)

        # Diagnóstico / animación
        self.gap_history: list[float] = []
        self.flow_snapshots: list[dict[tuple, float]] = []  # (u,v) -> flujo por iter

    # ── Función de costo BPR ─────────────────────────────────────────────────
    def _costs(self, flow: np.ndarray) -> np.ndarray:
        return self.t0 * (1.0 + self.alpha * (flow / self.cap) ** self.beta)

    # ── Asignación todo-o-nada dado un vector de costos ──────────────────────
    def _all_or_nothing(self, cost: np.ndarray) -> np.ndarray:
        """
        Carga toda la demanda O-D sobre los caminos mínimos según `cost`.
        Un Dijkstra single-source por centroide de origen.
        """
        # Publica el costo actual como atributo para que lo use Dijkstra.
        for i, (u, v, k) in enumerate(self.edge_list):
            self.graph[u][v][k]["_fw_cost"] = cost[i]

        y = np.zeros(len(self.edge_list), dtype=float)
        for oi, o in enumerate(self.nodes):
            _, paths = nx.single_source_dijkstra(self.graph, o, weight="_fw_cost")
            for dj, d in enumerate(self.nodes):
                if d == o:
                    continue
                demand = self.od[oi, dj]
                if demand <= 0.0:
                    continue
                path = paths.get(d)
                if not path or len(path) < 2:
                    continue
                for u, v in zip(path[:-1], path[1:]):
                    # arista paralela de menor costo entre u y v
                    best_i = min(self.uv_keys[(u, v)], key=lambda ki: cost[ki[1]])[1]
                    y[best_i] += demand
        return y

    # ── Búsqueda de línea exacta (bisección sobre el objetivo de Beckmann) ───

    def _line_search(self, x: np.ndarray, d: np.ndarray) -> float:
        """
        Paso lambda in [0, 1] que minimiza el objetivo de Beckmann a lo largo
        de x + lambda*d. La derivada direccional g(lambda) = <t(x+lambda*d), d>
        es monótona creciente (el costo crece con el flujo) -> bisección.
        """
        def g(lam: float) -> float:
            return float(np.dot(self._costs(x + lam * d), d))

        if g(0.0) >= 0.0:
            return 0.0
        if g(1.0) <= 0.0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # ── Agregación de flujo por par (u, v) ───────────────────────────────────
    def _aggregate(self, flow: np.ndarray) -> dict[tuple, float]:
        """Suma el flujo de aristas paralelas: (u, v, key) -> (u, v)."""
        agg: dict[tuple, float] = {}
        for i, (u, v, _) in enumerate(self.edge_list):
            f = flow[i]
            if f > 0.0:
                agg[(u, v)] = agg.get((u, v), 0.0) + f
        return agg

    # ── Bucle principal ──────────────────────────────────────────────────────
    def run(self) -> dict[tuple, float]:
        """Ejecuta Frank-Wolfe y devuelve el flujo de equilibrio {(u, v): flujo}."""
        if self.verbose:
            print(f"Frank-Wolfe: {len(self.edge_list):,} aristas, "
                  f"{len(self.nodes)} zonas, escala capacidad={self.capacity_scale:g}")

        # Iteración 0: todo-o-nada a costo free-flow.
        self.flow = self._all_or_nothing(self._costs(self.flow))
        self.flow_snapshots.append(self._aggregate(self.flow))

        for it in range(1, self.max_iter + 1):
            cost = self._costs(self.flow)
            y = self._all_or_nothing(cost)

            # Relative gap = TSTT / SPTT - 1  (0 en el equilibrio exacto).
            tstt = float(np.dot(cost, self.flow))   # tiempo total del sistema
            sptt = float(np.dot(cost, y))           # cota inferior (caminos mínimos)
            gap = (tstt / sptt - 1.0) if sptt > 0.0 else 0.0
            self.gap_history.append(gap)

            d = y - self.flow
            lam = self._line_search(self.flow, d)
            self.flow = self.flow + lam * d
            self.flow_snapshots.append(self._aggregate(self.flow))

            if self.verbose:
                print(f"  iter {it:2d}: gap={gap:.3e}  paso={lam:.4f}")
            if gap < self.tol:
                if self.verbose:
                    print(f"  Convergió en {it} iteraciones (gap < {self.tol:g}).")
                break
        else:
            if self.verbose:
                print(f"  Alcanzó max_iter={self.max_iter} sin bajar de tol.")

        self._cleanup()
        return self.edge_flows()

    def _cleanup(self) -> None:
        """Elimina el atributo temporal de costo para no ensuciar el grafo."""
        for u, v, k in self.edge_list:
            self.graph[u][v][k].pop("_fw_cost", None)

    # ── Resultados ───────────────────────────────────────────────────────────
    def edge_flows(self) -> dict[tuple, float]:
        """Flujo de equilibrio agregado por par {(u, v): flujo}."""
        return self._aggregate(self.flow)

    def vc_ratios(self) -> np.ndarray:
        """Cociente v/c por arista (índice de congestión relativa)."""
        return self.flow / self.cap

    def print_summary(self) -> None:
        vc = self.vc_ratios()
        active = self.flow > 0
        print("\n--- Equilibrio de Wardrop (Frank-Wolfe) ---")
        print(f"Iteraciones:              {len(self.gap_history)}")
        print(f"Relative gap final:       {self.gap_history[-1]:.3e}"
              if self.gap_history else "Relative gap final:       n/a")
        print(f"Aristas con flujo > 0:    {int(active.sum()):,}")
        if active.any():
            print(f"v/c mediano (activas):    {np.median(vc[active]):.2f}")
            print(f"v/c máximo:               {vc.max():.2f}")
            print(f"Aristas saturadas (v/c>1):{int((vc > 1.0).sum()):,}")