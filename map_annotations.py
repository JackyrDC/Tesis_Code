"""
Anotaciones compartidas para todos los mapas del pipeline: puntos de
interés (POIs) con su nombre y nombres de calle sobre la red vial.

Ambas funciones dibujan sobre un `ax` de matplotlib ya existente, así que
cualquier mapa (estático, animado o en vivo) puede llamarlas después de
dibujar la red. Los textos son artistas estáticos: en animaciones basta
con anotar UNA vez antes de arrancar los frames.

Supuesto de rotación: los ejes usan aspect="equal" sobre coordenadas
lon/lat (como todos los mapas del pipeline), de modo que el ángulo de una
arista en coordenadas de datos coincide con su ángulo en pantalla.
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np

# Color de marcador por categoría de POI: salud (rojo), educación (azul),
# comercio (verde). Categorías no mapeadas caen en morado.
POI_CATEGORY_COLORS: dict[str, str] = {
    "hospital": "#d62728", "clinic": "#d62728",
    "doctors": "#d62728", "pharmacy": "#d62728",
    "school": "#1f77b4", "college": "#1f77b4",
    "university": "#1f77b4", "kindergarten": "#1f77b4",
    "marketplace": "#2ca02c", "mall": "#2ca02c",
    "supermarket": "#2ca02c", "department_store": "#2ca02c",
}
_POI_DEFAULT_COLOR = "#9467bd"


def _halo(dark: bool) -> list:
    """Contorno alrededor del texto para que sea legible sobre la red."""
    return [pe.withStroke(linewidth=1.5, foreground="black" if dark else "white")]


class _ZoomTextScaler:
    """
    Reescala los textos anotados según el nivel de zoom del eje.

    El tamaño de fuente de matplotlib está en puntos (fijo respecto a la
    pantalla), así que al hacer zoom el mapa crece pero las etiquetas no.
    Este objeto escucha los cambios de límites del eje — que es lo que
    disparan el zoom y el pan de la toolbar — y ajusta cada texto en
    proporción a cuánto se acercó la vista respecto al encuadre inicial:
    nunca por debajo de su tamaño base (la vista completa se ve igual que
    siempre) y con un tope para que no ocupe media pantalla.
    """

    MAX_FONTSIZE = 15.0

    def __init__(self, ax):
        self.ax = ax
        self.items: list[tuple] = []      # (artista de texto, tamaño base)
        self.ref_width: float | None = None
        ax.callbacks.connect("xlim_changed", self._rescale)
        ax.callbacks.connect("ylim_changed", self._rescale)

    def register(self, texts, base_fontsize: float) -> None:
        self.items.extend((t, base_fontsize) for t in texts)
        if self.ref_width is None:
            xmin, xmax = self.ax.get_xlim()
            width = abs(xmax - xmin)
            self.ref_width = width if width > 0 else None

    def _rescale(self, ax) -> None:
        if not self.items or self.ref_width is None:
            return
        xmin, xmax = ax.get_xlim()
        width = abs(xmax - xmin)
        if width <= 0:
            return
        factor = self.ref_width / width
        for text, base in self.items:
            text.set_fontsize(min(max(base * factor, base), self.MAX_FONTSIZE))


def _zoom_scaler(ax) -> _ZoomTextScaler:
    """Un único escalador por eje, compartido entre POIs y nombres de calle."""
    scaler = getattr(ax, "_annotation_zoom_scaler", None)
    if scaler is None:
        scaler = _ZoomTextScaler(ax)
        ax._annotation_zoom_scaler = scaler
    return scaler


def annotate_pois(
    ax,
    pois: list[dict],
    fontsize: float = 5.0,
    dark: bool = False,
    marker_size: float = 16.0,
) -> None:
    """
    Dibuja cada POI como un triángulo coloreado por categoría y, si tiene
    nombre, lo etiqueta junto al marcador.

    Parameters
    ----------
    pois : lista de dicts {lat, lon, category, name, ...} — el formato de
           `ODMatrixBuilder.get_pois()`. Los POIs sin nombre en OSM se
           dibujan solo como marcador.
    dark : True si el fondo del mapa es oscuro (invierte texto y halo).
    """
    if not pois:
        return

    lons = [p["lon"] for p in pois]
    lats = [p["lat"] for p in pois]
    colors = [POI_CATEGORY_COLORS.get(p.get("category"), _POI_DEFAULT_COLOR) for p in pois]
    ax.scatter(
        lons, lats, s=marker_size, marker="^", c=colors,
        edgecolors="black" if dark else "white", linewidths=0.4, zorder=6,
    )

    text_color = "white" if dark else "black"
    labels = []
    for p in pois:
        name = p.get("name")
        if isinstance(name, str) and name.strip():
            labels.append(ax.annotate(
                name, xy=(p["lon"], p["lat"]),
                xytext=(2.5, 2.5), textcoords="offset points",
                fontsize=fontsize, color=text_color, zorder=7,
                path_effects=_halo(dark),
            ))
    _zoom_scaler(ax).register(labels, fontsize)


def annotate_street_names(
    ax,
    graph,
    fontsize: float = 4.0,
    dark: bool = False,
    min_length: float = 50.0,
) -> None:
    """
    Etiqueta cada calle UNA vez, sobre su arista más larga, con el texto
    rotado según la dirección de esa arista.

    Etiquetar cada arista individual repetiría el mismo nombre en cada
    cuadra (una calle de 10 cuadras son ~10 aristas); agrupar por nombre y
    quedarse con el segmento más largo es lo que hace cualquier mapa
    impreso. Si dos calles distintas comparten nombre, solo se etiqueta la
    de arista más larga — limitación aceptada a cambio de legibilidad.

    Parameters
    ----------
    min_length : longitud mínima (m) de la arista elegida para etiquetar;
                 filtra callejones donde el texto se saldría de la calle.
    dark       : True si el fondo del mapa es oscuro.
    """
    # Arista más larga por nombre de calle. `name` en OSM puede ser str,
    # lista (vías con varios nombres; se toma el primero) o faltar.
    best: dict[str, tuple[float, int, int]] = {}
    for u, v, data in graph.edges(data=True):
        name = data.get("name")
        if isinstance(name, (list, tuple)):
            name = name[0] if name else None
        if not isinstance(name, str) or not name.strip():
            continue
        length = float(data.get("length", 0.0))
        if name not in best or length > best[name][0]:
            best[name] = (length, u, v)

    text_color = "white" if dark else "black"
    labels = []
    for name, (length, u, v) in best.items():
        if length < min_length:
            continue
        x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        labels.append(ax.annotate(
            name, xy=((x1 + x2) / 2, (y1 + y2) / 2),
            rotation=angle, rotation_mode="anchor",
            ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=5,
            path_effects=_halo(dark),
        ))
    _zoom_scaler(ax).register(labels, fontsize)
