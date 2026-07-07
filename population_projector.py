

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neighborhood_data import BARRIOS_COMAYAGUA, TOTAL_POBLACION_URBANA

# ── Tipologías y tasas de crecimiento anual ────────────────────────────────────

TipologiaT = Literal["consolidado", "en_crecimiento", "emergente"]

# (tasa_baja, tasa_central, tasa_alta)  — fracción decimal, por año
_TASAS: dict[TipologiaT, tuple[float, float, float]] = {
    "consolidado":    (0.010, 0.015, 0.020),
    "en_crecimiento": (0.030, 0.0389, 0.048),
    "emergente":      (0.055, 0.075, 0.100),
}

# Barrios del centro histórico (consolidado): trazado colonial, expansión física limitada
_BARRIOS_CONSOLIDADOS: frozenset[str] = frozenset({
    "BARRIO ABAJO", "BARRIO ARRIBA", "BARRIO CABAÑAS", "BARRIO INDEPENDENCIA",
    "BARRIO LA CARIDAD", "BARRIO LA JOYA", "BARRIO LA ZARCITA",
    "BARRIO LOS LIRIOS", "BARRIO LOURDES",
    "BARRIO SAN ANTONIO DE LA SABANA", "BARRIO SAN BLAS", "BARRIO SAN FRANCISCO",
    "BARRIO SAN JOSE", "BARRIO SAN PABLO", "BARRIO SAN RAMON",
    "BARRIO SAN SEBASTIAN", "BARRIO SANTA LUCIA", "BARRIO SUYAPA",
    "BARRIO TORONDON", "BARRIO MATA DE CANA",
})


def _clasificar(nombre: str, poblacion_2022: int) -> TipologiaT:
    """Asigna tipología según nombre y población base."""
    if nombre in _BARRIOS_CONSOLIDADOS:
        return "consolidado"
    if poblacion_2022 < 500:
        return "emergente"
    return "en_crecimiento"


# ── Estructura de resultado ────────────────────────────────────────────────────

@dataclass
class ProyeccionBarrio:
    nombre: str
    tipologia: TipologiaT
    poblacion_2022: int
    poblacion_2025_baja: int
    poblacion_2025_central: int
    poblacion_2025_alta: int
    es_nuevo: bool = False          # True si proviene de OSM, no de Tabla 4

    @property
    def rango(self) -> str:
        return (f"{self.poblacion_2025_baja:,} – "
                f"{self.poblacion_2025_alta:,} "
                f"(central: {self.poblacion_2025_central:,})")


# ── Función principal de proyección ───────────────────────────────────────────

def proyectar_2025(
    anio_base: int = 2022,
    anio_objetivo: int = 2025,
) -> list[ProyeccionBarrio]:
    """
    Proyecta la población de cada barrio conocido de 2022 a 2025.
    Devuelve lista ordenada por población central descendente.
    """
    delta = anio_objetivo - anio_base
    resultados: list[ProyeccionBarrio] = []

    for nombre, datos in BARRIOS_COMAYAGUA.items():
        p22 = datos["poblacion"]
        tipo = _clasificar(nombre, p22)
        r_low, r_mid, r_high = _TASAS[tipo]

        resultados.append(ProyeccionBarrio(
            nombre=nombre,
            tipologia=tipo,
            poblacion_2022=p22,
            poblacion_2025_baja=int(round(p22 * (1 + r_low) ** delta)),
            poblacion_2025_central=int(round(p22 * (1 + r_mid) ** delta)),
            poblacion_2025_alta=int(round(p22 * (1 + r_high) ** delta)),
        ))

    resultados.sort(key=lambda x: x.poblacion_2025_central, reverse=True)
    return resultados


def poblacion_proyectada(
    escenario: Literal["baja", "central", "alta"] = "central",
    anio_objetivo: int = 2025,
) -> tuple[dict[str, int], int]:
    """
    Devuelve `(poblacion_por_barrio, total_urbana)` para el escenario indicado.

    Formato listo para alimentar `ODMatrixBuilder(population_by_zone=...,
    total_population=...)`.
    """
    resultados = proyectar_2025(anio_objetivo=anio_objetivo)
    attr = f"poblacion_2025_{escenario}"
    pop_por_barrio = {r.nombre: getattr(r, attr) for r in resultados}
    return pop_por_barrio, sum(pop_por_barrio.values())


# ── Estimación para asentamientos nuevos (IDW) ────────────────────────────────

def estimar_nuevo_asentamiento(
    lat: float,
    lon: float,
    area_m2: float,
    resultados_2025: list[ProyeccionBarrio],
    centroides: dict[str, tuple[float, float]],
    k: int = 5,
    power: float = 2.0,
) -> ProyeccionBarrio:
    """
    Estima población de un asentamiento nuevo (sin datos Tabla 4) via IDW.

    Parameters
    ----------
    lat, lon    : coordenadas del centroide del nuevo asentamiento
    area_m2     : área aproximada en m² (de polígono OSM u otro)
    resultados_2025 : lista ya proyectada de barrios conocidos
    centroides  : dict barrio → (lat, lon) de su centroide
    k           : k vecinos más cercanos para la interpolación
    power       : exponente de la distancia (default 2 = cuadrático)
    """
    # Densidad 2025 (hab/m²) por barrio conocido con centroide disponible
    pares: list[tuple[float, float]] = []   # (distancia_m, densidad)

    for proy in resultados_2025:
        if proy.nombre not in centroides:
            continue
        clat, clon = centroides[proy.nombre]
        # Distancia Haversine aproximada (suficiente a escala de ciudad)
        dlat = np.radians(lat - clat)
        dlon = np.radians(lon - clon)
        a = np.sin(dlat / 2) ** 2 + (
            np.cos(np.radians(clat)) * np.cos(np.radians(lat))
            * np.sin(dlon / 2) ** 2
        )
        dist_m = 6_371_000 * 2 * np.arcsin(np.sqrt(a))
        if dist_m < 1:
            dist_m = 1.0
        # Área del barrio vecino (aproximada por viviendas * 120 m² promedio)
        datos_base = BARRIOS_COMAYAGUA.get(proy.nombre, {})
        viv = datos_base.get("viviendas_ocupadas", 1) or 1
        area_barrio_m2 = viv * 120.0
        densidad = proy.poblacion_2025_central / area_barrio_m2
        pares.append((dist_m, densidad))

    if not pares:
        # Fallback: densidad municipal media (~4,100 hab/km²)
        densidad_idw = 4_100 / 1_000_000
    else:
        pares_k = sorted(pares, key=lambda x: x[0])[:k]
        pesos = [1 / (d ** power) for d, _ in pares_k]
        densidades = [dens for _, dens in pares_k]
        densidad_idw = sum(w * d for w, d in zip(pesos, densidades)) / sum(pesos)

    p_central = max(10, int(round(densidad_idw * area_m2)))

    # Intervalo ±30 % para asentamientos nuevos (incertidumbre mayor)
    return ProyeccionBarrio(
        nombre="NUEVO_ASENTAMIENTO",
        tipologia="emergente",
        poblacion_2022=0,
        poblacion_2025_baja=int(round(p_central * 0.70)),
        poblacion_2025_central=p_central,
        poblacion_2025_alta=int(round(p_central * 1.30)),
        es_nuevo=True,
    )


# ── Balance poblacional urbano ─────────────────────────────────────────────────

def resumen_balance(
    resultados: list[ProyeccionBarrio],
    tasa_municipal: float = 0.0389,
    fraccion_urbana: float = 0.6843,
    anio_base: int = 2022,
    anio_objetivo: int = 2025,
) -> dict:
    """
    Compara la suma de barrios proyectados con la proyección municipal total.
    El residual indica población urbana no capturada por la Tabla 4 (nuevas áreas).
    """
    delta = anio_objetivo - anio_base
    p_municipal_2022 = TOTAL_POBLACION_URBANA / fraccion_urbana   # total municipal
    p_municipal_obj  = p_municipal_2022 * (1 + tasa_municipal) ** delta
    p_urbana_obj     = p_municipal_obj * fraccion_urbana

    suma_baja    = sum(r.poblacion_2025_baja    for r in resultados if not r.es_nuevo)
    suma_central = sum(r.poblacion_2025_central for r in resultados if not r.es_nuevo)
    suma_alta    = sum(r.poblacion_2025_alta    for r in resultados if not r.es_nuevo)

    return {
        "poblacion_urbana_2022":      TOTAL_POBLACION_URBANA,
        "poblacion_urbana_2025_ref":  int(round(p_urbana_obj)),
        "suma_barrios_baja":          suma_baja,
        "suma_barrios_central":       suma_central,
        "suma_barrios_alta":          suma_alta,
        "residual_central":           int(round(p_urbana_obj - suma_central)),
        "cobertura_pct":              round(100 * suma_central / p_urbana_obj, 1),
    }


# ── CLI rápido ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resultados = proyectar_2025()
    balance    = resumen_balance(resultados)

    print("=" * 72)
    print(f"  PROYECCIÓN DEMOGRÁFICA 2022 → 2025  |  Comayagua (zona urbana)")
    print("=" * 72)
    print(f"{'Barrio/Colonia':<38} {'Tipo':<16} {'Bajo':>7} {'Central':>8} {'Alto':>7}")
    print("-" * 72)
    for r in resultados:
        tipo_short = {"consolidado": "Consol.", "en_crecimiento": "Crec.",
                      "emergente": "Emerg."}[r.tipologia]
        print(f"{r.nombre:<38} {tipo_short:<16} "
              f"{r.poblacion_2025_baja:>7,} {r.poblacion_2025_central:>8,} "
              f"{r.poblacion_2025_alta:>7,}")

    print("=" * 72)
    print(f"\n{'BALANCE POBLACIONAL URBANO':}")
    print(f"  Base 2022 (Tabla 4):              {balance['poblacion_urbana_2022']:>10,} hab.")
    print(f"  Referencia municipal 2025:        {balance['poblacion_urbana_2025_ref']:>10,} hab.")
    print(f"  Suma barrios conocidos (central): {balance['suma_barrios_central']:>10,} hab.")
    print(f"  Residual (nuevas áreas):          {balance['residual_central']:>10,} hab.")
    print(f"  Cobertura Tabla 4 → 2025:         {balance['cobertura_pct']:>9.1f} %")
    print()

    por_tipo: dict[str, list[int]] = {}
    for r in resultados:
        por_tipo.setdefault(r.tipologia, []).append(r.poblacion_2025_central)
    print("  Barrios por tipología:")
    for tipo, pops in por_tipo.items():
        print(f"    {tipo:<18}: {len(pops):>3} zonas  |  "
              f"pop. central total = {sum(pops):,}")
