"""
Comparación entre ejecuciones de config completa (topn=100, pop=200, gen=50):
daño dirigido (NSGA-II) contra fallos aleatorios (Monte Carlo) y frecuencia
de aristas críticas, una serie por semilla.

Descubre automáticamente las corridas válidas en `results/` y detecta
duplicados de random_failures.csv (el bug de propagación de semilla que
afectó a las corridas del 16 y 17 de julio: misma media/std/max Monte Carlo
pese a semillas distintas declaradas). Las corridas duplicadas se excluyen
del panel de aleatorios (no son un muestreo independiente real) pero se
conservan para el panel de aristas críticas, que no depende de esa semilla.

Uso:
    python compare_seeds.py
"""

import hashlib
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def _normalizar_calle(nombre):
    """
    Clave de comparación para una calle. Las aristas sin nombre en OSM se
    identifican por par de nodos, y ambos sentidos ("a-b" y "b-a") son la
    MISMA calle física — el greedy iterativo puede elegir el sentido
    contrario al que aparece en el ranking del NSGA-II. Se normalizan a un
    frozenset de los dos nodos para que cuenten como la misma arista.
    """
    m = re.match(r"\(sin nombre: (\d+)-(\d+)\)", nombre)
    if m:
        return frozenset(m.groups())
    return nombre

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_DIR = os.path.join(RESULTS_DIR, "comparacion_semillas")

# Config completa de referencia: solo corridas con estos parámetros son
# comparables entre sí (excluye ejecuciones de humo con pool/población chicos).
CONFIG_REFERENCIA = {
    "top_n_candidatos": 100,
    "poblacion_ga": 200,
    "generaciones_ga": 50,
}


def _leer_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _leer_csv_dicts(path):
    import csv
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def descubrir_todas_las_corridas():
    """
    TODAS las corridas con random_failures.csv, sin filtrar por config: el
    universo de aristas de random_failures.py es el grafo completo, no el
    pool del NSGA-II, así que el hash de duplicados debe compararse sobre
    el universo completo de corridas, no solo las de config completa.
    """
    corridas = []
    for nombre in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, nombre)
        meta_path = os.path.join(run_dir, "metadata.json")
        rf_path = os.path.join(run_dir, "random_failures.csv")
        if not (nombre.startswith("run_") and os.path.isfile(meta_path)
                and os.path.isfile(rf_path)):
            continue
        meta = _leer_json(meta_path)
        params = meta["parametros"]
        corridas.append({
            "run_id": nombre.removeprefix("run_"),
            "run_dir": run_dir,
            "semilla": params["semilla"],
            "es_config_referencia": all(
                params.get(k) == v for k, v in CONFIG_REFERENCIA.items()
            ),
            "meta": meta,
            "aleatorio": _leer_csv_dicts(rf_path),
        })
    return corridas


def _hash_archivo(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _marcar_duplicados_por(corridas, nombre_archivo, campo_flag, campo_origen):
    grupos = {}
    for c in corridas:
        h = _hash_archivo(os.path.join(c["run_dir"], nombre_archivo))
        grupos.setdefault(h, []).append(c)
    for grupo in grupos.values():
        grupo_ordenado = sorted(grupo, key=lambda c: c["run_id"])
        origen = grupo_ordenado[0]
        origen[campo_flag] = False
        for c in grupo_ordenado[1:]:
            c[campo_flag] = True
            c[campo_origen] = origen["run_id"]


def marcar_duplicados(corridas):
    """
    Dos chequeos de duplicado independientes, sobre TODAS las corridas (no
    solo las de config completa):

    - `duplicado_ga`: hash de `pareto_front_by_generation.csv` idéntico.
      Significa que la corrida COMPLETA del NSGA-II —no solo el Monte
      Carlo— es una copia de otra con semilla declarada distinta. Invalida
      la corrida para CUALQUIER comparación entre semillas, no solo la de
      aleatorios.
    - `duplicado`: hash de `random_failures.csv` idéntico, sin que el
      NSGA-II también lo esté — solo invalida el panel de fallos aleatorios.

    En ambos casos la más antigua globalmente queda como "origen" (no
    duplicada) y el resto se marca. Si `duplicado_ga` es cierto, `duplicado`
    también se fuerza a cierto: una corrida cuyo NSGA-II entero es una copia
    no puede tratarse como independiente para nada.
    """
    _marcar_duplicados_por(corridas, "random_failures.csv", "duplicado", "duplicado_de")
    _marcar_duplicados_por(corridas, "pareto_front_by_generation.csv",
                            "duplicado_ga", "duplicado_ga_de")
    for c in corridas:
        if c.get("duplicado_ga"):
            c["duplicado"] = True
            c["duplicado_de"] = c.get("duplicado_de") or c["duplicado_ga_de"]


def _cargar_hv_por_generacion(run_dir):
    """{generación: hipervolumen} a partir de pareto_front_by_generation.csv
    (el HV se repite por cada solución de la generación; basta un valor)."""
    path = os.path.join(run_dir, "pareto_front_by_generation.csv")
    hv_por_gen = {}
    for fila in _leer_csv_dicts(path):
        hv_por_gen[int(fila["generacion"])] = float(fila["hipervolumen"])
    return hv_por_gen


def graficar_convergencia_hv(corridas, extra, ax):
    """
    Hipervolumen por generación superpuesto entre todas las ejecuciones de
    config completa disponibles (más las de config distinta como
    referencia cruzada) — llena el hueco que la propia tesis declara en
    "Alcance y limitaciones": la estabilidad del frente ante semillas
    distintas no se había medido.
    """
    validas = [c for c in corridas if not c.get("duplicado_ga")]
    excluidas = [c for c in corridas if c.get("duplicado_ga")]
    todas = validas + list(extra)
    cmap = plt.colormaps["tab10"]

    convergencias = []
    for i, c in enumerate(todas):
        hv_por_gen = _cargar_hv_por_generacion(c["run_dir"])
        gens = sorted(hv_por_gen)
        hvs = [hv_por_gen[g] for g in gens]
        hv_final = hvs[-1]

        # primera generación en tocar el HV final (dentro de una tolerancia
        # mínima por redondeo de punto flotante).
        gen_convergencia = next(g for g, v in zip(gens, hvs) if v >= hv_final - 0.01)

        es_extra = c in extra
        estilo = dict(linestyle="--", alpha=0.75) if es_extra else dict(alpha=0.9)
        ax.plot(gens, hvs, color=cmap(i), lw=1.8,
                label=f"{_etiqueta_config(c, CONFIG_REFERENCIA)} (converge en gen. {gen_convergencia})",
                **estilo)
        ax.scatter([gen_convergencia], [hv_final], color=cmap(i), s=45, zorder=5,
                   marker="D" if es_extra else "o", edgecolors="white", linewidths=0.8)
        convergencias.append((c["semilla"], gen_convergencia, es_extra))

    ax.set_xlabel("Generación")
    ax.set_ylabel("Hipervolumen del frente")
    ax.set_title("Convergencia del hipervolumen: comparación entre semillas")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    rango = [g for _, g, extra_f in convergencias if not extra_f]
    nota = ""
    if rango:
        nota = (f"Generación de convergencia entre semillas de la config de referencia: "
                f"{min(rango)}–{max(rango)} (de 50 generaciones configuradas).")
    if excluidas:
        detalle_exc = ", ".join(f"semilla {c['semilla']}" for c in excluidas)
        nota += (f"\nNo incluida(s) por ser NSGA-II duplicado byte-a-byte de otra corrida "
                 f"(mismo pareto_front_by_generation.csv pese a semilla declarada distinta): {detalle_exc}.")
    if nota:
        ax.text(0.01, -0.16, nota, transform=ax.transAxes, fontsize=8, color="#555555")


def _etiqueta_config(c, referencia):
    """Nombre corto para la leyenda; marca la config si difiere de la de referencia."""
    p = c["meta"]["parametros"]
    if all(p.get(k) == v for k, v in referencia.items()):
        return f"semilla {c['semilla']}"
    return (f"semilla {c['semilla']} — config distinta "
            f"(top_n={p['top_n_candidatos']}, fw={p['fw_max_iter']})")


def graficar_dano_por_metodo(corridas, extra, ax):
    """
    ΔTTV real (no razón) por método: NSGA-II, las dos voraces y el fallo
    aleatorio, agrupados por tamaño de corte. NSGA-II y greedy apenas
    cambian entre ejecuciones válidas (se toman de una corrida de
    referencia); el aleatorio sí varía por semilla, así que se muestra su
    media entre ejecuciones válidas con una barra de error al mínimo/máximo
    observado. Escala log en el eje: el aleatorio es 2-3 órdenes de
    magnitud más chico que los otros tres métodos y en lineal desaparecería.
    """
    cortes = [1, 2, 3, 4, 5]
    validas = [c for c in corridas if not c["duplicado"]]
    excluidas = [c for c in corridas if c["duplicado"]]
    aleatorias = validas + list(extra)

    referencia = validas[0]
    greedy = _leer_csv_dicts(os.path.join(referencia["run_dir"], "greedy_baseline.csv"))
    dirigido = {int(r["num_cortes"]): float(r["dirigido"]) for r in referencia["aleatorio"]}
    greedy_flujo = {int(r["num_cortes"]): float(r["greedy_flujo"]) for r in greedy}
    greedy_iter = {int(r["num_cortes"]): float(r["greedy_iterativo"]) for r in greedy}

    aleatorio_por_k = {k: [] for k in cortes}
    for c in aleatorias:
        medias = {int(r["num_cortes"]): float(r["aleatorio_media"]) for r in c["aleatorio"]}
        for k in cortes:
            aleatorio_por_k[k].append(medias[k])

    metodos = [
        ("NSGA-II (dirigido)", [dirigido[k] for k in cortes], None, "#1f4e79"),
        ("Greedy por flujo", [greedy_flujo[k] for k in cortes], None, "#cc5500"),
        ("Greedy iterativo", [greedy_iter[k] for k in cortes], None, "#2e8b57"),
        ("Aleatorio (media entre semillas)",
         [float(np.mean(aleatorio_por_k[k])) for k in cortes],
         [[float(np.mean(aleatorio_por_k[k]) - min(aleatorio_por_k[k])) for k in cortes],
          [float(max(aleatorio_por_k[k]) - np.mean(aleatorio_por_k[k])) for k in cortes]],
         "#999999"),
    ]

    x = np.arange(len(cortes))
    ancho = 0.8 / len(metodos)

    for i, (nombre, valores, yerr, color) in enumerate(metodos):
        barras = ax.bar(x + i * ancho, valores, width=ancho, color=color,
                         yerr=yerr, capsize=3, label=nombre)
        ax.bar_label(barras, fmt=lambda v: f"{v:,.0f}", fontsize=6.5, rotation=90, padding=3)

    ax.set_yscale("log")
    ax.set_xlabel("Número de aristas cortadas")
    ax.set_ylabel("ΔTTV (veh·s/día, escala log)")
    ax.set_title("Daño real por método, por tamaño de corte")
    ax.set_xticks(x + ancho * (len(metodos) - 1) / 2)
    ax.set_xticklabels(cortes)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # espacio para las etiquetas
    ax.grid(True, axis="y", alpha=0.3, which="major")
    ax.legend(fontsize=8)

    n_semillas = len(aleatorias)
    nota = (f"Aleatorio: media y rango (mín-máx) entre {n_semillas} ejecuciones con Monte "
            f"Carlo independiente (semillas {', '.join(str(c['semilla']) for c in aleatorias)}). "
            f"NSGA-II y greedy tomados de la corrida semilla {referencia['semilla']} "
            f"(prácticamente iguales en todas las ejecuciones válidas).")
    if excluidas:
        nota += (" No incluidas por Monte Carlo duplicado: "
                 + ", ".join(f"semilla {c['semilla']}" for c in excluidas) + ".")
    ax.text(0.01, -0.16, nota, transform=ax.transAxes, fontsize=7, color="#555555",
            ha="left", wrap=True)


def _greedy_frecuencia_por_calle(run_dir, columna, calles_top_norm):
    """
    % de los tamaños de corte evaluados (k=1..5) en los que la estrategia
    greedy (`columna` = calles_greedy_flujo o calles_greedy_iterativo)
    incluye cada una de las calles de `calles_top_norm` (ya normalizadas).
    """
    filas = _leer_csv_dicts(os.path.join(run_dir, "greedy_baseline.csv"))
    if not filas:
        return {c: 0.0 for c in calles_top_norm}
    conteo = {c: 0 for c in calles_top_norm}
    for fila in filas:
        presentes = {_normalizar_calle(n.strip()) for n in fila[columna].split("|")}
        for calle_norm in calles_top_norm:
            if calle_norm in presentes:
                conteo[calle_norm] += 1
    return {c: 100.0 * n / len(filas) for c, n in conteo.items()}


def graficar_aristas_criticas(corridas, extra, ax, top_n=5):
    # Se excluyen las corridas con NSGA-II duplicado byte-a-byte de otra:
    # no aportan una semilla independiente, mostrarlas infla artificialmente el acuerdo
    # entre "semillas". Las de config distinta (extra) se muestran aparte.
    corridas_ga_validas = [c for c in corridas if not c.get("duplicado_ga")]
    excluidas_ga = [c for c in corridas if c.get("duplicado_ga")]
    todas_nsga = corridas_ga_validas + list(extra)
    conteo_por_calle = {}
    for c in todas_nsga:
        tam_frente = c["meta"]["resultados"]["tamano_frente_final"]
        for entrada in c["meta"]["resultados"]["aristas_criticas_top"]:
            calle_norm = _normalizar_calle(entrada["calle"])
            pct = 100.0 * entrada["frecuencia_en_frente"] / tam_frente
            conteo_por_calle.setdefault(calle_norm, {"nombre": entrada["calle"], "pcts": []})
            conteo_por_calle[calle_norm]["pcts"].append(pct)

    calles_top_norm = sorted(conteo_por_calle,
                              key=lambda k: -np.mean(conteo_por_calle[k]["pcts"]))[:top_n]
    etiquetas = [conteo_por_calle[c]["nombre"] for c in calles_top_norm]

    referencia = next(c for c in corridas if not c["duplicado"])
    greedy_flujo_pct = _greedy_frecuencia_por_calle(
        referencia["run_dir"], "calles_greedy_flujo", calles_top_norm)
    greedy_iter_pct = _greedy_frecuencia_por_calle(
        referencia["run_dir"], "calles_greedy_iterativo", calles_top_norm)

    series = todas_nsga + ["greedy_flujo", "greedy_iterativo"]
    x = np.arange(len(calles_top_norm))
    ancho = 0.8 / len(series)
    cmap = plt.colormaps["tab10"]

    for i, c in enumerate(todas_nsga):
        por_calle = {_normalizar_calle(e["calle"]): e["frecuencia_en_frente"]
                     for e in c["meta"]["resultados"]["aristas_criticas_top"]}
        tam_frente = c["meta"]["resultados"]["tamano_frente_final"]
        pcts = [100.0 * por_calle.get(cn, 0) / tam_frente for cn in calles_top_norm]
        es_extra = c in extra
        ax.bar(x + i * ancho, pcts, width=ancho, color=cmap(i),
               hatch="//" if es_extra else None,
               label=f"NSGA-II, {_etiqueta_config(c, CONFIG_REFERENCIA)}")

    ax.bar(x + len(todas_nsga) * ancho, [greedy_flujo_pct[c] for c in calles_top_norm],
           width=ancho, color="#cc5500", label="Greedy por flujo")
    ax.bar(x + (len(todas_nsga) + 1) * ancho, [greedy_iter_pct[c] for c in calles_top_norm],
           width=ancho, color="#2e8b57", label="Greedy iterativo")

    ax.set_xticks(x + ancho * (len(series) - 1) / 2)
    ax.set_xticklabels(etiquetas, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% de escenarios en que aparece\n(NSGA-II: % del frente final; greedy: % de los 5 tamaños de corte)")
    ax.set_title(f"Top {top_n} aristas críticas: NSGA-II vs. greedy")
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)

    if excluidas_ga:
        detalle = ", ".join(f"semilla {c['semilla']}" for c in excluidas_ga)
        ax.text(0.01, -0.42, f"No incluida(s) por ser NSGA-II duplicado byte-a-byte de otra "
                f"corrida: {detalle}.", transform=ax.transAxes, fontsize=7.5, color="#555555")


def main():
    todas = descubrir_todas_las_corridas()
    if not todas:
        print("No se encontraron corridas con random_failures.csv.")
        return
    marcar_duplicados(todas)  # sobre TODO el universo, no solo config completa

    corridas = [c for c in todas if c["es_config_referencia"]]
    if not corridas:
        print("No hay corridas de config completa (topn=100, pop=200, gen=50).")
        return

    validas = [c for c in corridas if not c["duplicado"]]
    excluidas = [c for c in corridas if c["duplicado"]]
    print(f"Corridas de config completa encontradas: {len(corridas)}")
    print(f"  válidas para el panel aleatorio: {[c['semilla'] for c in validas]}")
    if excluidas:
        detalle = [(c["semilla"], c.get("duplicado_de") or "otra config") for c in excluidas]
        print(f"  excluidas (Monte Carlo duplicado): {detalle}")

    otras_duplicadas = [c for c in todas if c["duplicado"] and not c["es_config_referencia"]]
    if otras_duplicadas:
        print(f"  (aviso: el mismo problema de duplicado también aparece fuera de la "
              f"config de referencia: {[c['run_id'] for c in otras_duplicadas]})")

    # Corridas fuera de la config de referencia pero con Monte Carlo válido
    # (origen no-duplicado) y misma escala de población (200): aportan un
    # punto de comparación extra aunque su pool/fw difieran. Las de escala
    # mucho menor (p. ej. pop=50) no se agregan: su "dirigido" no es
    # comparable en la misma escala del eje.
    extra = [
        c for c in todas
        if not c["es_config_referencia"] and not c["duplicado"]
        and c["meta"]["parametros"].get("poblacion_ga") == CONFIG_REFERENCIA["poblacion_ga"]
    ]
    if extra:
        print(f"  + agregadas como referencia cruzada (otra config, Monte Carlo propio): "
              f"{[(c['semilla'], c['run_id']) for c in extra]}")

    if not validas:
        print("Ninguna corrida de config completa tiene Monte Carlo verificadamente "
              "independiente. Abortando el gráfico.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(10, 6.5))
    graficar_dano_por_metodo(corridas, extra, ax1)
    fig1.tight_layout()
    path1 = os.path.join(OUT_DIR, "dano_por_metodo.png")
    fig1.savefig(path1, dpi=200)
    print(f"\nGráfico 1 guardado en {path1}")

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    graficar_aristas_criticas(corridas, extra, ax2)
    fig2.tight_layout()
    path2 = os.path.join(OUT_DIR, "aristas_criticas_por_semilla.png")
    fig2.savefig(path2, dpi=200)
    print(f"Gráfico 2 guardado en {path2}")

    fig3, ax3 = plt.subplots(figsize=(9.5, 6))
    graficar_convergencia_hv(corridas, extra, ax3)
    fig3.tight_layout()
    path3 = os.path.join(OUT_DIR, "convergencia_hv_por_semilla.png")
    fig3.savefig(path3, dpi=200)
    print(f"Gráfico 3 guardado en {path3}")


if __name__ == "__main__":
    main()
