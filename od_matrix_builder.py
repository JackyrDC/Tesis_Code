import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as osm
import pandas as pd

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree

from neighborhood_data import BARRIOS_COMAYAGUA, TOTAL_POBLACION_URBANA

# Archivo de caché para evitar re-geocodificar en cada ejecución
_CENTROID_CACHE = os.path.join(os.path.dirname(__file__), "cache", "barrio_centroids.json")

# Archivo de caché para no repetir la descarga de puntos de interés (Overpass)
_POI_CACHE = os.path.join(os.path.dirname(__file__), "cache", "poi_locations.json")

# Etiquetas OSM de puntos de interés que actúan como polos de atracción de
# viajes (salud, educación y comercio) — clave = tag OSM, valores = a buscar.
_POI_TAGS: dict[str, list[str]] = {
    "amenity": [
        "hospital", "clinic", "doctors", "pharmacy",
        "marketplace", "school", "college", "university", "kindergarten",
    ],
    "shop": ["mall", "supermarket", "department_store"],
}

# Peso relativo de atracción por categoría (proxy de generación de viajes;
# valores relativos entre categorías, no absolutos — ajustables según
# literatura de generación de viajes si se dispone de datos locales).
_POI_ATTRACTION_WEIGHT: dict[str, float] = {
    "hospital": 5.0, "clinic": 2.0, "doctors": 1.5, "pharmacy": 1.0,
    "marketplace": 4.0, "mall": 4.5, "supermarket": 3.0, "department_store": 3.5,
    "school": 2.5, "college": 3.0, "university": 3.5, "kindergarten": 1.5,
}

# Bounding box aproximado del casco urbano de Comayagua (lat_min, lat_max, lon_min, lon_max)
_COMAYAGUA_BBOX: tuple[float, float, float, float] = (14.35, 14.55, -87.72, -87.55)

# Expansiones de prefijos usados en el catálogo (`BO.`, `COL.`, etc.).
# Se emplean para generar variantes de query en la geocodificación
# — Nominatim suele reconocer nombres completos ("Colonia Los Pinos")
# pero no las abreviaturas ("COL. LOS PINOS").
_PREFIX_EXPANSIONS: dict[str, str] = {
    "COL.":   "Colonia",
    "BO.":    "Barrio",
    "BARRIO": "Barrio",
    "RES.":   "Residencial",
    "LOT.":   "Lotificación",
    "TR.":    "",   # significado incierto — se prueba sin prefijo
}


def _in_comayagua_bbox(lat: float, lon: float) -> bool:
    lat_min, lat_max, lon_min, lon_max = _COMAYAGUA_BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _query_variants(barrio_name: str) -> list[str]:
    """
    Devuelve variantes de query para Nominatim, en orden de preferencia:
      1. Nombre original tal como está.
      2. Con el prefijo expandido ("COL." → "Colonia").
      3. Sin el prefijo (solo el nombre propio).
    Todas se sufijan con ", Comayagua, Honduras" para desambiguar.
    """
    tokens = barrio_name.split()
    prefix = tokens[0] if tokens else ""
    tail   = " ".join(tokens[1:]) if len(tokens) > 1 else barrio_name

    variants = [f"{barrio_name}, Comayagua, Honduras"]

    if prefix in _PREFIX_EXPANSIONS:
        expanded = _PREFIX_EXPANSIONS[prefix]
        if expanded:
            variants.append(f"{expanded} {tail}, Comayagua, Honduras")

    if len(tokens) > 1 and tail:
        variants.append(f"{tail}, Comayagua, Honduras")

    # Dedup preservando orden
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


class ODMatrixBuilder:
    """
    Construye la matriz Origen-Destino del grafo vial de Comayagua.

    Ponderación híbrida:
      - Componente sociodemográfico: población por barrio/colonia (UNAH 2022).
      - Componente geoespacial: asignación espacial de nodos OSM a barrios
        mediante polígonos OSM (primer intento) o geocodificación de centroides
        con árbol KD (segundo intento).
      - Componente topológico: grado de salida de cada nodo (fallback).
      - Componente de atracción por puntos de interés: hospitales, mercados,
        centros comerciales, escuelas y colegios cercanos a cada nodo, usado
        para ponderar el DESTINO de los viajes (los orígenes siguen pesados
        solo por población/topología, ya que representan "salidas de casa").

    El parámetro `alpha` (0-1) controla el balance demográfico/topológico
    del peso de origen. El parámetro `poi_gamma` (0-1) controla cuánto del
    peso de destino viene de la atracción por POIs vs. el peso de origen.
    """

    def __init__(
        self,
        graph,
        total_demand: int | None = None,
        use_demographics: bool = True,
        alpha: float = 1.0,
        place_name: str = "Comayagua, Honduras",
        trip_rate: float = 2.0,
        use_poi_attraction: bool = True,
        poi_gamma: float = 0.4,
        population_by_zone: dict[str, int] | None = None,
        total_population: int | None = None,
    ):
        self.graph = self._ensure_travel_times(graph)
        self.population_by_zone = (
            dict(population_by_zone) if population_by_zone is not None
            else {name: data["poblacion"] for name, data in BARRIOS_COMAYAGUA.items()}
        )
        self.total_population = (
            int(total_population) if total_population is not None
            else TOTAL_POBLACION_URBANA
        )
        if total_demand is None:
            total_demand = self.total_population
        self.total_demand = int(total_demand * trip_rate)
        self.trip_rate = trip_rate
        self.use_demographics = use_demographics
        self.alpha = alpha
        self.place_name = place_name
        self.use_poi_attraction = use_poi_attraction
        self.poi_gamma = poi_gamma

        self.nodes = list(self.graph.nodes())
        self.n = len(self.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}

        self._weights: np.ndarray | None = None
        self._dest_weights: np.ndarray | None = None
        self._poi_weights: np.ndarray | None = None
        self._pois: list[dict] | None = None
        self._travel_times: np.ndarray | None = None
        self._paths: dict[int, dict[int, list]] | None = None
        self._od_matrix: np.ndarray | None = None
        self._zone_assignments: dict | None = None  # node_id -> barrio_name

    # ── Infraestructura ─────────────────────────────────────────────────────────

    def _ensure_travel_times(self, graph):
        sample = next(iter(graph.edges(data=True)), (None, None, {}))[2]
        if "speed_kph" not in sample:
            graph = osm.add_edge_speeds(graph)
        if "travel_time" not in sample:
            graph = osm.add_edge_travel_times(graph)
        return graph

    def _node_latlon(self) -> np.ndarray:
        """Devuelve array (n, 2) con (lat, lon) de cada nodo."""
        return np.array(
            [(self.graph.nodes[n]["y"], self.graph.nodes[n]["x"]) for n in self.nodes]
        )

    # ── Asignación espacial nodo → barrio ───────────────────────────────────────

    def _load_centroid_cache(self) -> dict:
        if os.path.exists(_CENTROID_CACHE):
            with open(_CENTROID_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_centroid_cache(self, cache: dict) -> None:
        os.makedirs(os.path.dirname(_CENTROID_CACHE), exist_ok=True)
        with open(_CENTROID_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def _geocode_centroids(self) -> dict[str, list[float]]:
        """
        Geocodifica cada barrio para obtener su centroide (lat, lon).

        Estrategia:
          1. Cachea localmente para no repetir llamadas a Nominatim.
          2. Purga entradas fuera del bounding box de Comayagua (errores de
             homonimia con otros municipios) para reintentarlas.
          3. Para cada barrio faltante, prueba varias variantes de query
             (nombre original, prefijo expandido, sin prefijo).
          4. Solo acepta resultados dentro del bounding box de Comayagua.

        Devuelve solo los barrios que se geocodificaron con éxito.
        """
        cache = self._load_centroid_cache()
        changed = False

        # Purga: entradas fuera de bbox son claramente errores de Nominatim
        # (nombres homónimos en otros municipios). Se reintentan abajo.
        out_of_bbox = [
            name for name, coord in cache.items()
            if not _in_comayagua_bbox(coord[0], coord[1])
        ]
        if out_of_bbox:
            print(f"  Descartando {len(out_of_bbox)} centroides fuera de Comayagua "
                  f"(homonimia): {', '.join(out_of_bbox)}")
            for name in out_of_bbox:
                del cache[name]
            changed = True

        for name in BARRIOS_COMAYAGUA:
            if name in cache:
                continue
            found: tuple[float, float] | None = None
            for query in _query_variants(name):
                try:
                    lat, lon = osm.geocode(query)
                except Exception:
                    continue
                if _in_comayagua_bbox(lat, lon):
                    found = (lat, lon)
                    print(f"  [geocode OK] {name}  ← {query!r}")
                    break
            if found:
                cache[name] = [found[0], found[1]]
                changed = True
            else:
                print(f"  [geocode FAIL] {name}  (probó {len(_query_variants(name))} variantes)")

        if changed:
            self._save_centroid_cache(cache)

        aun_faltan = [n for n in BARRIOS_COMAYAGUA if n not in cache]
        if aun_faltan:
            print(f"  {len(aun_faltan)} barrios sin centroide (Nominatim no los encontró); "
                  f"sus nodos se asignarán al centroide más cercano por Voronoi:")
            for n in aun_faltan:
                print(f"    · {n}")

        # Solo devolver los que están en nuestro catálogo
        return {k: v for k, v in cache.items() if k in BARRIOS_COMAYAGUA}

    # ── Puntos de interés (atracción de destino) ────────────────────────────────

    def _load_poi_cache(self) -> list[dict] | None:
        if os.path.exists(_POI_CACHE):
            with open(_POI_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_poi_cache(self, pois: list[dict]) -> None:
        os.makedirs(os.path.dirname(_POI_CACHE), exist_ok=True)
        with open(_POI_CACHE, "w", encoding="utf-8") as f:
            json.dump(pois, f, ensure_ascii=False, indent=2)

    def _fetch_pois(self) -> list[dict]:
        """
        Descarga (o recupera de caché) los puntos de interés relevantes:
        hospitales, clínicas, farmacias, mercados, centros comerciales,
        escuelas, colegios y universidades. Devuelve una lista de dicts
        {lat, lon, category, weight}.
        """
        if self._pois is not None:
            return self._pois

        cached = self._load_poi_cache()
        if cached is not None:
            self._pois = cached
            return self._pois

        print("Descargando puntos de interés (salud, educación, comercio) desde OSM...")
        try:
            gdf = osm.features_from_place(self.place_name, tags=_POI_TAGS)
        except Exception as exc:
            print(f"  [POI FAIL] No se pudieron descargar POIs: {exc}")
            self._pois = []
            return self._pois

        pois: list[dict] = []
        for _, row in gdf.iterrows():
            category = None
            for key in ("amenity", "shop"):
                val = row.get(key)
                if isinstance(val, str) and val in _POI_ATTRACTION_WEIGHT:
                    category = val
                    break
            if category is None:
                continue
            try:
                centroid = row.geometry.centroid
            except Exception:
                continue
            pois.append({
                "lat": float(centroid.y),
                "lon": float(centroid.x),
                "category": category,
                "weight": _POI_ATTRACTION_WEIGHT[category],
            })

        self._save_poi_cache(pois)
        print(f"  {len(pois)} POIs descargados y cacheados.")
        self._pois = pois
        return self._pois

    def _poi_attraction_weights(self) -> np.ndarray:
        """
        Peso de atracción por nodo: cada POI se asigna al nodo de red más
        cercano (árbol KD sobre lat/lon, igual que la asignación de zonas)
        y aporta el peso de su categoría. Se normaliza para sumar 1.
        """
        pois = self._fetch_pois()
        weights = np.zeros(self.n)

        if not pois:
            print("Sin POIs disponibles; el destino se ponderará solo por demografía.")
            return weights

        tree = KDTree(self._node_latlon())
        poi_coords = np.array([(p["lat"], p["lon"]) for p in pois])
        _, idx = tree.query(poi_coords)

        for i_node, poi in zip(idx, pois):
            weights[i_node] += poi["weight"]

        total = weights.sum()
        if total > 0:
            weights /= total
        self._poi_weights = weights
        return weights

    def _try_osm_polygons(self) -> dict[str, object]:
        """
        Intenta recuperar polígonos de barrios desde OSM.
        Devuelve dict: nombre_normalizado -> geometry (Shapely).
        """
        try:
            gdf = osm.features_from_place(
                self.place_name,
                tags={"place": ["neighbourhood", "suburb", "quarter"]},
            )
            if gdf.empty:
                return {}
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            result = {}
            for _, row in gdf.iterrows():
                name = str(row.get("name", "")).strip().upper()
                if name:
                    result[name] = row.geometry
            return result
        except Exception:
            return {}

    def _match_osm_to_catalog(self, osm_names: list[str]) -> dict[str, str]:
        """Empareja nombres de polígonos OSM con el catálogo de barrios (fuzzy)."""
        catalog_names = list(BARRIOS_COMAYAGUA.keys())
        osm_to_catalog: dict[str, str] = {}
        for osm_name in osm_names:
            for cat_name in catalog_names:
                if osm_name == cat_name.upper() or osm_name in cat_name.upper():
                    osm_to_catalog[osm_name] = cat_name
                    break
        return osm_to_catalog

    def assign_nodes_to_zones(self) -> dict:
        """
        Asigna cada nodo del grafo al barrio/colonia más cercano.

        Estrategia en cascada:
          1. Polígonos OSM (place=neighbourhood/suburb/quarter): asignación
             punto-en-polígono para nodos cubiertos.
          2. Geocodificación de centroides + árbol KD: nodos restantes.
          3. Si todo falla, devuelve dict vacío (se usarán pesos topológicos).
        """
        if self._zone_assignments is not None:
            return self._zone_assignments

        coords = self._node_latlon()              # (n, 2)  [lat, lon]
        assignments: dict = {}

        # ── Paso 1: polígonos OSM ──────────────────────────────────────────────
        print("Buscando polígonos de barrios en OSM...")
        osm_polygons = self._try_osm_polygons()
        if osm_polygons:
            from shapely.geometry import Point as SPoint

            osm_to_catalog = self._match_osm_to_catalog(list(osm_polygons.keys()))

            if osm_to_catalog:
                print(f"  {len(osm_to_catalog)} barrios con polígono en OSM.")
                for i, node in enumerate(self.nodes):
                    pt = SPoint(coords[i][1], coords[i][0])   # (lon, lat)
                    for osm_name, geom in osm_polygons.items():
                        if osm_name in osm_to_catalog and geom.contains(pt):
                            assignments[node] = osm_to_catalog[osm_name]
                            break
                covered = len(assignments)
                print(f"  {covered}/{self.n} nodos asignados por polígonos OSM.")

        # ── Paso 2: geocodificación + KD-tree para nodos sin asignar ──────────
        # Voronoi crudo (nodo → centroide más cercano). Se probó una variante
        # ponderada por √población para que barrios grandes reclamaran más área,
        # pero terminó diluyendo los pesos de origen/destino en los barrios
        # populosos y aplanando la matriz O-D, con la consecuencia visible de
        # que las concentraciones de flujo en las arterias desaparecían. Volver
        # a Voronoi puro preserva la compactitud geográfica de cada zona.
        unassigned = [node for node in self.nodes if node not in assignments]
        if unassigned:
            print(f"Geocodificando centroides para {len(unassigned)} nodos sin zona...")
            centroids = self._geocode_centroids()

            if centroids:
                cent_names = list(centroids.keys())
                cent_coords = np.array([centroids[n] for n in cent_names])  # (m, 2)

                unassigned_coords = np.array(
                    [(self.graph.nodes[n]["y"], self.graph.nodes[n]["x"])
                     for n in unassigned]
                )
                tree = KDTree(cent_coords)
                _, idx = tree.query(unassigned_coords)

                for node, zone_idx in zip(unassigned, idx):
                    assignments[node] = cent_names[zone_idx]

                print(f"  {len(unassigned)} nodos asignados por geocodificación+KD-tree "
                      f"({len(cent_names)} centroides).")
            else:
                print("  Geocodificación fallida — se usarán pesos topológicos.")

        self._zone_assignments = assignments
        zones_used = len(set(assignments.values()))
        print(f"Asignación completada: {len(assignments)}/{self.n} nodos → {zones_used} zonas.")
        return self._zone_assignments

    # ── Mapa de zonas (barrios) ─────────────────────────────────────────────────

    def plot_zone_map(
        self,
        figsize: tuple[float, float] = (13, 11),
        cmap: str = "gist_ncar",
        label_zones: bool = True,
        label_fontsize: float = 4.5,
    ):
        """
        Mapa intermedio: la ciudad segmentada por barrios/colonias.

        A diferencia de una versión anterior basada en polígonos de OSM
        (cobertura parcial), este mapa colorea directamente la red vial
        según la asignación nodo→zona de `assign_nodes_to_zones()` — la
        misma que alimenta los pesos demográficos de la matriz O-D — así
        que cubre todos los nodos (polígono OSM si existe, si no la zona
        del centroide geocodificado más cercano) y es visualmente
        consistente con lo que el modelo realmente usa.

        Usa `osmnx.plot_graph` con `node_color`/`edge_color` por nodo
        (un color por barrio, de `osmnx.plot.get_colors`) y etiqueta cada
        zona en el centroide de sus nodos.
        """
        zone_map = self.assign_nodes_to_zones()
        if not zone_map:
            print("No hay asignación de zonas disponible; no se puede graficar el mapa.")
            return None

        names = sorted(set(zone_map.values()))
        palette = osm.plot.get_colors(len(names), cmap=cmap)
        zone_color = dict(zip(names, palette))
        default_color = "#cccccc"  # nodos sin zona asignada (no debería ocurrir)

        node_color = [zone_color.get(zone_map.get(n), default_color) for n in self.nodes]
        edge_color = [
            zone_color.get(zone_map.get(u), default_color)
            for u, _, _ in self.graph.edges(keys=True)
        ]

        fig, ax = osm.plot_graph(
            self.graph, figsize=figsize,
            node_color=node_color, node_size=6, node_edgecolor="none", node_zorder=2,
            edge_color=edge_color, edge_linewidth=1.3,
            bgcolor="white", show=False, close=False, save=False,
        )

        if label_zones:
            coords_by_zone: dict[str, list[tuple[float, float]]] = {}
            for node, zone in zone_map.items():
                coords_by_zone.setdefault(zone, []).append(
                    (self.graph.nodes[node]["x"], self.graph.nodes[node]["y"])
                )
            for zone, pts in coords_by_zone.items():
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                ax.annotate(
                    zone.title(), xy=(cx, cy), fontsize=label_fontsize,
                    ha="center", va="center", color="black", weight="bold", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65),
                )

        assigned, total_zones = len(zone_map), len(names)
        ax.set_title(
            f"Comayagua segmentada por barrios/colonias (asignación de nodos)\n"
            f"{assigned}/{self.n} nodos asignados a {total_zones} zonas",
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()
        return fig, ax

    # ── Cómputo de pesos ────────────────────────────────────────────────────────

    def _topological_weights(self) -> np.ndarray:
        """Pesos basados en grado de salida (comportamiento original)."""
        degrees = np.array([self.graph.out_degree(n) for n in self.nodes], dtype=float)
        total = degrees.sum()
        return degrees / total if total > 0 else np.ones(self.n) / self.n

    def _demographic_weights(self) -> np.ndarray:
        """
        Pesos demográficos: cada nodo recibe una fracción de la población
        de su barrio proporcional al número de nodos en ese barrio.

        w_i = P(zona_i) / (|nodos en zona_i| * P_total_urbana)
        """
        zone_map = self.assign_nodes_to_zones()

        if not zone_map:
            print("Sin asignación de zonas; usando pesos topológicos.")
            return self._topological_weights()

        # Conteo de nodos por zona
        zone_count: dict[str, int] = {}
        for zone in zone_map.values():
            zone_count[zone] = zone_count.get(zone, 0) + 1

        weights = np.zeros(self.n)
        fallback_weight = 1.0 / self.n  # para nodos sin zona válida

        for i, node in enumerate(self.nodes):
            zone = zone_map.get(node)
            if zone and zone in self.population_by_zone:
                pop = self.population_by_zone[zone]
                count = zone_count.get(zone, 1)
                weights[i] = pop / (count * self.total_population)
            else:
                weights[i] = fallback_weight

        # Normalizar para que sumen 1
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    def compute_node_weights(self) -> np.ndarray:
        """
        Calcula los pesos finales de cada nodo:
          - use_demographics=False → puramente topológico (grado de salida).
          - use_demographics=True  → mezcla ponderada:
              w = alpha * w_demo + (1 - alpha) * w_topo
        """
        topo = self._topological_weights()

        if not self.use_demographics:
            self._weights = topo
            return self._weights

        demo = self._demographic_weights()
        blended = self.alpha * demo + (1.0 - self.alpha) * topo

        total = blended.sum()
        self._weights = blended / total if total > 0 else blended
        return self._weights

    def compute_destination_weights(self) -> np.ndarray:
        """
        Calcula el peso de DESTINO de cada nodo, distinto del peso de origen:
        mezcla la atracción por puntos de interés (hospitales, mercados,
        centros comerciales, escuelas, colegios) con el peso de origen
        (población/topología), que sigue aportando una atracción base
        genérica (p. ej. visitas entre viviendas).

          w_dest = poi_gamma * w_poi + (1 - poi_gamma) * w_origen

        Si `use_poi_attraction=False` o no hay POIs disponibles, el destino
        es idéntico al origen (comportamiento previo).
        """
        if self._weights is None:
            self.compute_node_weights()

        if not self.use_poi_attraction:
            self._dest_weights = self._weights
            return self._dest_weights

        poi = self._poi_attraction_weights()
        if poi.sum() == 0:
            self._dest_weights = self._weights
            return self._dest_weights

        blended = self.poi_gamma * poi + (1.0 - self.poi_gamma) * self._weights
        total = blended.sum()
        self._dest_weights = blended / total if total > 0 else blended
        return self._dest_weights

    # ── Tiempos de viaje ────────────────────────────────────────────────────────

    def compute_travel_times(self) -> np.ndarray:
        """
        Calcula tiempos mínimos Y caminos mínimos en un único pase all-pairs
        Dijkstra (las rutas se guardan para reutilizarlas en BaselineBuilder
        sin recalcular la misma operación costosa dos veces).
        """
        print(f"Calculando caminos mínimos para {self.n} nodos...")
        T = np.full((self.n, self.n), np.inf)
        paths: dict[int, dict[int, list]] = {}
        for u, (lengths, node_paths) in nx.all_pairs_dijkstra(self.graph, weight="travel_time"):
            paths[u] = node_paths
            i = self.node_index[u]
            for v, length in lengths.items():
                j = self.node_index.get(v)
                if j is not None:
                    T[i, j] = length
        np.fill_diagonal(T, 0.0)
        self._travel_times = T
        self._paths = paths
        return T

    def get_paths(self) -> dict[int, dict[int, list]]:
        """Devuelve dict origen -> {destino: camino_mas_corto} (reutilizable)."""
        if self._paths is None:
            self.compute_travel_times()
        return self._paths

    # ── Construcción de la matriz O-D ───────────────────────────────────────────

    def build_od_matrix(self, beta: float = 0.01) -> np.ndarray:
        """
        Construye la matriz O-D usando el modelo gravitacional:

          OD_ij = w_i * w_j^dest * exp(-beta * t_ij)

        donde w_i es el peso de origen (demográfico+geoespacial) del nodo i,
        w_j^dest es el peso de destino (POI + demográfico) del nodo j, y
        t_ij es el tiempo de viaje mínimo (segundos) entre i y j.

        La matriz se escala para que la demanda total sea `self.total_demand`.
        """
        if self._weights is None:
            self.compute_node_weights()
        if self._dest_weights is None:
            self.compute_destination_weights()
        if self._travel_times is None:
            self.compute_travel_times()

        impedance = np.where(np.isinf(self._travel_times), 0.0,
                             np.exp(-beta * self._travel_times))

        od = np.outer(self._weights, self._dest_weights) * impedance
        np.fill_diagonal(od, 0.0)

        total = od.sum()
        if total > 0:
            od *= self.total_demand / total

        self._od_matrix = od
        return od

    # ── Accesores ───────────────────────────────────────────────────────────────

    def get_matrix(self) -> np.ndarray:
        if self._od_matrix is None:
            self.build_od_matrix()
        return self._od_matrix

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_matrix(), index=self.nodes, columns=self.nodes)

    def get_travel_times_dataframe(self) -> pd.DataFrame:
        if self._travel_times is None:
            self.compute_travel_times()
        return pd.DataFrame(self._travel_times, index=self.nodes, columns=self.nodes)

    def get_zone_assignment_series(self) -> pd.Series:
        """Devuelve Series: node_id → barrio asignado."""
        assignments = self.assign_nodes_to_zones()
        return pd.Series(
            {n: assignments.get(n, "SIN_ZONA") for n in self.nodes},
            name="barrio",
        )

    # ── Resumen ─────────────────────────────────────────────────────────────────

    def print_summary(self):
        od = self.get_matrix()
        T = self._travel_times

        reachable = int(np.sum(np.isfinite(T) & (T > 0)))
        total_possible = self.n * (self.n - 1)

        print("\n--- Resumen Matriz O-D (ponderada) ---")
        print(f"Nodos en la red:           {self.n}")
        print(f"Ponderación demográfica:   {'Sí' if self.use_demographics else 'No'}"
              f"  (alpha={self.alpha:.2f})")
        print(f"Población urbana total:    {self.total_population:,} hab.")
        print(f"Tasa de viajes/persona:    {self.trip_rate:.1f} viajes/día")
        print(f"Demanda total escalada:    {od.sum():.1f} viajes")
        print(f"Pares O-D alcanzables:     {reachable} / {total_possible} "
              f"({100 * reachable / total_possible:.1f}%)")
        print(f"Flujo máximo:              {od.max():.4f}")
        i, j = np.unravel_index(np.argmax(od), od.shape)
        print(f"  Par de mayor flujo:      {self.nodes[i]} → {self.nodes[j]}")
        print(f"  Tiempo de viaje:         {T[i, j]:.1f} s ({T[i, j] / 60:.1f} min)")
        print(f"Flujo promedio (alcanz.):  {od[od > 0].mean():.6f}")

        # Top 5 barrios por peso demográfico asignado
        if self.use_demographics and self._zone_assignments:
            zone_weights: dict[str, float] = {}
            for i_node, node in enumerate(self.nodes):
                zone = self._zone_assignments.get(node, "SIN_ZONA")
                zone_weights[zone] = zone_weights.get(zone, 0.0) + float(self._weights[i_node])
            top5 = sorted(zone_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 zonas por peso en la red:")
            for rank, (zone, w) in enumerate(top5, 1):
                pop = self.population_by_zone.get(zone, 0)
                print(f"  {rank}. {zone:<35} peso={w:.4f}  (pop={pop:,})")

        # Resumen de puntos de interés (atracción de destino)
        if self.use_poi_attraction and self._pois:
            by_category: dict[str, int] = {}
            for p in self._pois:
                by_category[p["category"]] = by_category.get(p["category"], 0) + 1
            print(f"\nPuntos de interés (gamma={self.poi_gamma:.2f}): "
                  f"{len(self._pois)} encontrados")
            for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
                print(f"  {cat:<20} {count:>4}")

            if self._poi_weights is not None and self._zone_assignments:
                zone_poi: dict[str, float] = {}
                for i_node, node in enumerate(self.nodes):
                    zone = self._zone_assignments.get(node, "SIN_ZONA")
                    zone_poi[zone] = zone_poi.get(zone, 0.0) + float(self._poi_weights[i_node])
                top5_poi = sorted(zone_poi.items(), key=lambda x: x[1], reverse=True)[:5]
                print("\nTop 5 zonas por atracción POI (destino):")
                for rank, (zone, w) in enumerate(top5_poi, 1):
                    print(f"  {rank}. {zone:<35} atracción={w:.4f}")
