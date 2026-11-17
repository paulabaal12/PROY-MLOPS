# Databricks notebook source
# MAGIC %md
# MAGIC # Pronóstico v2: serie sin saltos falsos + variables externas (FRED)
# MAGIC
# MAGIC Versión **de prueba** de `capa_gold`. **No modifica** ninguna tabla existente: escribe solo tablas nuevas con sufijo `_v2`.
# MAGIC
# MAGIC **Qué cambia respecto a `capa_gold`**
# MAGIC
# MAGIC 1. **Paso 1 – serie sin saltos falsos.** La variación semanal (`dif_q`) y la etiqueta `direccion` solo se calculan entre semanas
# MAGIC    consecutivas de la **misma familia de fuente** (foto→foto, MAGA→MAGA). Antes, una semana con foto comparada contra una semana
# MAGIC    MAGA interpolada producía saltos de Q3–Q5 que eran cambio de fuente y no cambio de mercado.
# MAGIC 2. **Paso 4 – variables externas.** Se agregan cambios semanales del precio internacional (FRED: gasolina y diésel de la Costa del Golfo, WTI)
# MAGIC    y el subsidio ("Apoyo Social Temporal").
# MAGIC 3. **Evaluación honesta.** Validación *rolling-origin* (se reentrena cada N semanas, sin futuro), comparada contra dos líneas base
# MAGIC    (repetir dirección y predecir siempre la clase más frecuente) con intervalo de confianza (bootstrap por bloques de semanas).
# MAGIC    Se reporta por separado qué parte de la evaluación usa semanas **reales** (foto) y cuál usa semanas **interpoladas** de MAGA.
# MAGIC
# MAGIC **Supuesto pendiente:** la fecha en que terminó el subsidio no la tenemos. Se controla con el widget `subsidio_fin` (vacío = sigue vigente).

# COMMAND ----------

def _widget(nombre, defecto, etiqueta):
    try:
        dbutils.widgets.text(nombre, defecto, etiqueta)
        return dbutils.widgets.get(nombre)
    except NameError:                       # fuera de Databricks (pruebas locales)
        return defecto

CATALOGO       = _widget("catalogo", "equipo4", "Catálogo")
ESQUEMA_SALIDA = _widget("esquema_salida", "gold", "Esquema donde se escriben las tablas _v2")
UMBRAL_Q       = float(_widget("umbral_direccion", "0.10", "Umbral sube/baja (Q)"))
SEMANAS_PRUEBA = int(_widget("semanas_prueba", "52", "Semanas de prueba (las más recientes con etiqueta)"))
PASO_REENTRENO = int(_widget("paso_reentreno", "13", "Reentrenar cada N semanas dentro de la prueba"))
TIPO_CAMBIO    = float(_widget("tipo_cambio", "7.7", "GTQ por USD (constante)"))
SUBSIDIO_INI   = _widget("subsidio_ini", "2026-04-28", "Inicio del subsidio (AAAA-MM-DD)")
SUBSIDIO_FIN   = _widget("subsidio_fin", "", "Fin del subsidio (vacío = sigue vigente)")
RUTA_FRED      = _widget("ruta_fred", f"/Volumes/equipo4/bronze/archivos/fred", "Carpeta con CSV de FRED si no hay internet")

# COMMAND ----------

# ==== FUNCIONES ====
import numpy as np
import pandas as pd

OFICIALES     = ["diesel", "regular", "super"]
ORIGENES_FOTO = ["foto", "foto_sin_referencia", "corregido_manual"]
CLASES        = ["baja", "se_mantiene", "sube"]
SERIE_FRED    = {"DGASUSGULF": "gas_gulf", "DDFUELUSGULF": "diesel_gulf", "DCOILWTICO": "wti"}
FRED_DE       = {"diesel": "diesel_gulf", "regular": "gas_gulf", "super": "gas_gulf"}

def _ns(s):
    return pd.to_datetime(s).astype("datetime64[ns]")

def inicio_semana(fechas):
    """Martes de la semana (martes a lunes, como el MEM)."""
    f = pd.Series(_ns(fechas)).dt.normalize()
    return f - pd.to_timedelta((f.dt.day_of_week - 1) % 7, unit="D")

def familia(fuente):
    """foto / maga / arrastre: solo se comparan semanas de la misma familia."""
    if fuente is None or (isinstance(fuente, float) and np.isnan(fuente)):
        return None
    return "maga" if str(fuente).startswith("maga") else str(fuente)

def fotos_por_semana(lect):
    partes = []
    for c in OFICIALES:
        d = lect[["archivo", "fecha_hora", c, f"{c}_origen"]].rename(columns={c: "precio", f"{c}_origen": "origen"})
        d = d[d["origen"].isin(ORIGENES_FOTO) & d["precio"].notna() & d["fecha_hora"].notna()].copy()
        d["combustible"] = c
        partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    d["semana_inicio"] = inicio_semana(d["fecha_hora"]).to_numpy()
    return (d.groupby(["combustible", "semana_inicio"])
             .agg(promedio=("precio", "mean"), n_lecturas=("precio", "size")).reset_index())

def maga_por_semana(ofi, semanas):
    """Interpolación lineal entre promedios mensuales (puestos a mitad de mes). Igual que capa_gold."""
    medio = _ns(semanas + pd.Timedelta(days=3))
    filas = []
    for c in OFICIALES:
        s = ofi[ofi["combustible"] == c].sort_values("mes_inicio")
        if s.empty:
            continue
        ini, fin = s["mes_inicio"].min(), s["mes_inicio"].max() + pd.offsets.MonthEnd(0)
        dentro = ((medio >= ini) & (medio <= fin)).to_numpy()
        x = _ns(s["mes_inicio"] + pd.Timedelta(days=14)).astype("int64").to_numpy()
        precio = np.interp(medio.astype("int64").to_numpy(), x, s["precio_oficial"].to_numpy())
        filas.append(pd.DataFrame({"combustible": c, "semana_inicio": _ns(semanas).to_numpy(),
                                   "precio_maga": np.where(dentro, precio, np.nan)}))
    return pd.concat(filas, ignore_index=True).dropna(subset=["precio_maga"])

def construir_precio_semanal(lect, ofi):
    fotos = fotos_por_semana(lect)
    ini = min(fotos["semana_inicio"].min(), inicio_semana(pd.Series([ofi["mes_inicio"].min()])).iloc[0])
    fin = max(fotos["semana_inicio"].max(),
              inicio_semana(pd.Series([ofi["mes_inicio"].max() + pd.offsets.MonthEnd(0)])).iloc[0])
    semanas = pd.Series(pd.date_range(ini, fin, freq="7D")).astype("datetime64[ns]")
    maga = maga_por_semana(ofi, semanas)
    base = pd.MultiIndex.from_product([OFICIALES, semanas], names=["combustible", "semana_inicio"]).to_frame(index=False)
    d = (base.merge(fotos, on=["combustible", "semana_inicio"], how="left")
             .merge(maga, on=["combustible", "semana_inicio"], how="left")
             .sort_values(["combustible", "semana_inicio"]))
    hay_foto = d["promedio"].notna()
    d["fuente"] = np.where(hay_foto, "foto", np.where(d["precio_maga"].notna(), "maga_interpolado", None))
    d["promedio"] = d["promedio"].where(hay_foto, d["precio_maga"])
    sin_dato = d["promedio"].isna()
    d["promedio"] = d.groupby("combustible")["promedio"].ffill()
    d.loc[sin_dato & d["promedio"].notna(), "fuente"] = "arrastre"
    d = d[d["promedio"].notna()].copy()
    d["n_lecturas"] = d["n_lecturas"].fillna(0).astype(int)
    d["promedio"] = d["promedio"].round(2)
    return d[["combustible", "semana_inicio", "promedio", "n_lecturas", "fuente"]].reset_index(drop=True)

def direccion(dif, umbral):
    return pd.Series(np.select([dif > umbral, dif < -umbral], ["sube", "baja"], "se_mantiene"),
                     index=dif.index).where(dif.notna())

def construir_variacion(precio, umbral):
    """PASO 1: dif_q solo entre semanas consecutivas (7 días) de la MISMA familia de fuente."""
    d = precio.sort_values(["combustible", "semana_inicio"]).copy()
    g = d.groupby("combustible")
    d["precio_anterior"] = g["promedio"].shift(1)
    d["fuente_anterior"] = g["fuente"].shift(1)
    d["semana_anterior"] = g["semana_inicio"].shift(1)
    d["familia"] = d["fuente"].map(familia)
    d["familia_anterior"] = d["fuente_anterior"].map(familia)
    mismo = ((d["familia"] == d["familia_anterior"]) & ((d["semana_inicio"] - d["semana_anterior"]).dt.days == 7)
             & (d["familia"] != "arrastre"))
    d["dif_q"] = (d["promedio"] - d["precio_anterior"]).where(mismo).round(2)
    d["dif_pct"] = (100 * d["dif_q"] / d["precio_anterior"]).round(2)
    d["direccion"] = direccion(d["dif_q"], umbral)
    return d.drop(columns=["semana_anterior"]).reset_index(drop=True)

# ---- Variables externas (paso 4) ----
def fred_semanal(series, tipo_cambio):
    """series: dict {id_FRED: DataFrame(observation_date, valor)}. Devuelve promedio semanal (martes-lunes).
    gasolina y diésel se pasan a Q/galón; WTI queda en USD/barril."""
    partes = []
    for sid, nombre in SERIE_FRED.items():
        df = series[sid].copy()
        df.columns = ["fecha", "valor"]
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna()
        df["semana_inicio"] = inicio_semana(df["fecha"]).to_numpy()
        s = df.groupby("semana_inicio")["valor"].mean()
        if nombre != "wti":
            s = s * tipo_cambio
        partes.append(s.rename(nombre))
    return pd.concat(partes, axis=1).sort_index().rename_axis("semana_inicio").reset_index()

def subsidio_por_semana(semanas, ini, fin):
    s = pd.Series(_ns(semanas))
    vigente = (s >= pd.Timestamp(ini)) & ((s <= pd.Timestamp(fin)) if fin else True)
    out = pd.DataFrame({"semana_inicio": s, "subsidio_vigente": vigente.astype(float)})
    out["subsidio_cambio"] = out["subsidio_vigente"].diff().fillna(0.0)     # +1 inicia, -1 termina
    return out

FEATURES_BASE = ["dif_1", "dif_2", "dif_3", "dif_4", "dif_pct_1", "desvio_media_4", "dif_acum_4",
                 "spread_super_diesel", "spread_regular_diesel", "dir_actual", "es_diesel", "es_regular", "es_super"]
FEATURES_EXO  = ["fred_dif_1", "fred_dif_2", "fred_dif_4", "fred_dif_8", "wti_dif_4", "subsidio_vigente", "subsidio_cambio"]

def construir_features(variacion, fred, subsidio):
    d = variacion.sort_values(["combustible", "semana_inicio"]).copy()
    g = d.groupby("combustible")
    d["dif_1"] = d["dif_q"]
    for k in (2, 3, 4):
        d[f"dif_{k}"] = g["dif_q"].shift(k - 1)
    d["dif_pct_1"] = d["dif_pct"]
    d["media_movil_4"] = g["promedio"].transform(lambda s: s.rolling(4, min_periods=4).mean()).round(4)
    d["desvio_media_4"] = (d["promedio"] - d["media_movil_4"]).round(4)
    d["dif_acum_4"] = (d["promedio"] - g["promedio"].shift(4)).round(4)
    ancho = d.pivot(index="semana_inicio", columns="combustible", values="promedio")
    sp = pd.DataFrame({"semana_inicio": ancho.index,
                       "spread_super_diesel": (ancho["super"] - ancho["diesel"]).to_numpy(),
                       "spread_regular_diesel": (ancho["regular"] - ancho["diesel"]).to_numpy()})
    d = d.merge(sp, on="semana_inicio", how="left")
    d["dir_actual"] = d["direccion"].map({"baja": -1.0, "se_mantiene": 0.0, "sube": 1.0})
    for c in OFICIALES:
        d[f"es_{c}"] = (d["combustible"] == c).astype(float)
    # --- exógenas: cambios del precio internacional en la semana (ya conocidas al cierre de la semana base) ---
    f = fred.sort_values("semana_inicio").copy()
    for nombre in ("gas_gulf", "diesel_gulf"):
        for k in (1, 2, 4, 8):
            f[f"{nombre}_d{k}"] = f[nombre].diff(k)
        f[f"{nombre}_d2"] = f[nombre].diff(1).shift(1)          # cambio de la semana anterior
    f["wti_dif_4"] = f["wti"].diff(4)
    d = d.merge(f[["semana_inicio", "wti_dif_4"] + [c for c in f.columns if c.endswith(("_d1", "_d2", "_d4", "_d8"))]],
                on="semana_inicio", how="left")
    for k, nombre in ((1, "fred_dif_1"), (2, "fred_dif_2"), (4, "fred_dif_4"), (8, "fred_dif_8")):
        col = "_d2" if k == 2 else f"_d{k}"
        d[nombre] = np.where(d["combustible"] == "diesel", d[f"diesel_gulf{col}"], d[f"gas_gulf{col}"])
    d = d.drop(columns=[c for c in d.columns if c.startswith(("gas_gulf_", "diesel_gulf_"))])
    d = d.merge(subsidio, on="semana_inicio", how="left")
    g = d.groupby("combustible")
    d["direccion_siguiente"] = g["direccion"].shift(-1)
    d["fuente_siguiente"] = g["fuente"].shift(-1)
    d["familia_siguiente"] = d["fuente_siguiente"].map(familia)
    # Fila apta: semana, anterior y siguiente observadas, misma familia, sin arrastre, con todas las variables base
    d["apta_modelo"] = (d["direccion"].notna() & d["direccion_siguiente"].notna() & (d["familia"] == d["familia_siguiente"])
                        & d[FEATURES_BASE].notna().all(axis=1))
    d["apta_exo"] = d["apta_modelo"] & d[FEATURES_EXO].notna().all(axis=1)
    return d.reset_index(drop=True)

# ---- Evaluación ----
def macro_f1(real, pred):
    from sklearn.metrics import f1_score
    return f1_score(real, pred, average="macro", labels=CLASES, zero_division=0)

def evaluar(feats, modelos, semanas_prueba, paso, usar_col="apta_modelo", filtro_familia=None):
    """Rolling-origin: para cada bloque de `paso` semanas de prueba se reentrena con filas de semanas anteriores
    (semana_inicio < inicio del bloque) y se predice el bloque. modelos: {nombre: (constructor, columnas)}.
    Línea base 'repetir' = dirección actual; 'moda' = clase más frecuente del entrenamiento."""
    from sklearn.base import clone
    e = feats[feats[usar_col]].copy()
    if filtro_familia:
        e = e[e["familia"] == filtro_familia]
    semanas = np.sort(e["semana_inicio"].unique())
    if len(semanas) < 20:
        return None
    prueba_sem = semanas[-min(semanas_prueba, int(len(semanas) * 0.3)):]
    salidas = []
    for i in range(0, len(prueba_sem), paso):
        bloque = prueba_sem[i:i + paso]
        ini = bloque[0]
        tr, te = e[e["semana_inicio"] < ini], e[e["semana_inicio"].isin(bloque)]
        if tr["direccion_siguiente"].nunique() < 2 or te.empty:
            continue
        out = te[["combustible", "semana_inicio", "direccion_siguiente"]].copy()
        out["repetir"] = te["direccion"].to_numpy()
        out["moda"] = tr["direccion_siguiente"].mode().iloc[0]
        for nombre, (mk, cols) in modelos.items():
            m = clone(mk).fit(tr[cols], tr["direccion_siguiente"])
            out[nombre] = m.predict(te[cols])
        salidas.append(out)
    return pd.concat(salidas, ignore_index=True) if salidas else None

def resumen(pred, columnas, base="repetir", n_boot=2000, bloque=4, semilla=0):
    """Accuracy y F1 por modelo + IC 95% de la diferencia de accuracy contra la base (bootstrap por bloques de semanas)."""
    rng = np.random.default_rng(semilla)
    real = pred["direccion_siguiente"].to_numpy()
    sem = np.sort(pred["semana_inicio"].unique())
    idx_sem = {s: np.where(pred["semana_inicio"].to_numpy() == s)[0] for s in sem}
    n_b = max(1, len(sem) // bloque)
    filas = []
    for c in columnas:
        acierto = (pred[c].to_numpy() == real).astype(float)
        ac_base = (pred[base].to_numpy() == real).astype(float)
        difs = []
        for _ in range(n_boot):
            ini = rng.integers(0, max(1, len(sem) - bloque + 1), n_b)
            ii = np.concatenate([idx_sem[s] for j in ini for s in sem[j:j + bloque]])
            difs.append(acierto[ii].mean() - ac_base[ii].mean())
        lo, hi = np.percentile(difs, [2.5, 97.5])
        filas.append({"modelo": c, "n": len(pred), "semanas": len(sem), "accuracy": acierto.mean(),
                      "f1_macro": macro_f1(real, pred[c]), "dif_vs_" + base: acierto.mean() - ac_base.mean(),
                      "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(filas)
# ==== FIN FUNCIONES ====

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leer silver y FRED

# COMMAND ----------

import os
ESCRIBIR = _widget("escribir", "no", "Escribir tablas _v2 (si / no)").lower() == "si"

lect_cols = ["archivo", "fecha_hora"] + [x for c in OFICIALES for x in (c, f"{c}_origen")]
lect = spark.table(f"{CATALOGO}.silver.lecturas_fotos").select(*lect_cols).toPandas()
ofi  = spark.table(f"{CATALOGO}.silver.precios_oficiales_mensuales").select("combustible", "mes_inicio", "precio_oficial").toPandas()
lect["fecha_hora"] = pd.to_datetime(lect["fecha_hora"])
if lect["fecha_hora"].dt.tz is not None:
    lect["fecha_hora"] = lect["fecha_hora"].dt.tz_convert(None)
ofi["mes_inicio"] = pd.to_datetime(ofi["mes_inicio"])
ofi["precio_oficial"] = ofi["precio_oficial"].astype(float)
for c in OFICIALES:
    lect[c] = pd.to_numeric(lect[c], errors="coerce")
print(f"silver: {len(lect)} fotos, {len(ofi)} precios mensuales")

def leer_fred(sid):
    try:
        return pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    except Exception as e:                       # sin internet en el workspace
        ruta = f"{RUTA_FRED}/{sid}.csv"
        if os.path.exists(ruta):
            return pd.read_csv(ruta)
        raise RuntimeError(f"No se pudo descargar {sid} ({e}). Descárgalo de FRED y súbelo a {ruta}")

fred_csv = {sid: leer_fred(sid) for sid in SERIE_FRED}
print({sid: f"{d.iloc[:, 0].min()} a {d.iloc[:, 0].max()}" for sid, d in fred_csv.items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Serie corregida (paso 1) y variables externas (paso 4)

# COMMAND ----------

precio   = construir_precio_semanal(lect, ofi)
variacion = construir_variacion(precio, UMBRAL_Q)
fred_sem = fred_semanal(fred_csv, TIPO_CAMBIO)
subsidio = subsidio_por_semana(precio["semana_inicio"].unique(), SUBSIDIO_INI, SUBSIDIO_FIN or None)
feats    = construir_features(variacion, fred_sem, subsidio)

print(f"filas: {len(feats)}  | aptas para modelar: {int(feats['apta_exo'].sum())}")
print("aptas por familia de fuente:", feats[feats["apta_exo"]].groupby("familia").size().to_dict())
print("semanas foto→foto consecutivas (las únicas 'reales' evaluables):",
      int(((feats["familia"] == "foto") & feats["apta_exo"]).sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluación semanal (rolling-origin) con intervalos de confianza
# MAGIC
# MAGIC Lectura: `dif_vs_repetir` es la diferencia de accuracy contra repetir la dirección anterior; si el intervalo `ic95_inf`–`ic95_sup`
# MAGIC incluye 0, **no hay evidencia de mejora**. Importante: en semanas MAGA interpoladas la dirección casi siempre se repite
# MAGIC (la interpolación lineal lo fuerza), así que `repetir` tiene accuracy alta sin aprender nada.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
GB = lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
modelos = {"logit_base": (LR(), FEATURES_BASE), "logit_exo": (LR(), FEATURES_BASE + FEATURES_EXO),
           "gb_base": (GB(), FEATURES_BASE),    "gb_exo": (GB(), FEATURES_BASE + FEATURES_EXO)}

resultados = []
for nombre, fam, n_sem in (("maga_52s", "maga", SEMANAS_PRUEBA), ("maga_104s", "maga", 2 * SEMANAS_PRUEBA), ("foto", "foto", 200)):
    pr = evaluar(feats, modelos, n_sem, PASO_REENTRENO, usar_col="apta_exo", filtro_familia=fam)
    if pr is None:
        print(f"[{nombre}] muy pocas semanas para evaluar")
        continue
    r = resumen(pr, ["repetir", "moda"] + list(modelos)).assign(escenario=nombre)
    resultados.append(r)
    print(f"\n=== {nombre} ==="); print(r.round(3).to_string(index=False))
metricas_sem = pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Diagnóstico mensual (serie oficial MAGA, sin interpolar)
# MAGIC
# MAGIC Los 252 precios mensuales son el único dato oficial real. Aquí el objetivo es la dirección del **mes siguiente**, con cambios mensuales
# MAGIC propios y del precio internacional (FRED). Se prueban dos umbrales; con 81 casos de prueba y 2 umbrales, tómalo como indicio.

# COMMAND ----------

def datos_mensuales(ofi, fred_csv, tipo_cambio, umbral):
    fm = {}
    for sid, n in SERIE_FRED.items():
        d = fred_csv[sid].copy(); d.columns = ["f", "v"]
        d["v"] = pd.to_numeric(d["v"], errors="coerce"); d["f"] = pd.to_datetime(d["f"])
        fm[n] = d.dropna().set_index("f")["v"].resample("MS").mean() * (tipo_cambio if n != "wti" else 1.0)
    fm = pd.DataFrame(fm)
    filas = []
    for c in OFICIALES:
        s = ofi[ofi["combustible"] == c].sort_values("mes_inicio").set_index("mes_inicio")["precio_oficial"].asfreq("MS")
        d = pd.DataFrame({"p": s})
        d["dif_1"] = d["p"].diff(); d["dif_2"] = d["dif_1"].shift(1); d["dif_3"] = d["dif_1"].shift(2)
        fx = fm["diesel_gulf"] if c == "diesel" else fm["gas_gulf"]
        d["fred_1"] = fx.diff().reindex(d.index); d["fred_2"] = fx.diff().shift(1).reindex(d.index)
        d["fred_3"] = fx.diff(3).reindex(d.index); d["wti_1"] = fm["wti"].diff().reindex(d.index)
        d["direccion"] = direccion(d["dif_1"], umbral); d["direccion_siguiente"] = d["direccion"].shift(-1)
        d["combustible"] = c; d["semana_inicio"] = d.index
        for k in OFICIALES:
            d[f"es_{k}"] = float(c == k)
        filas.append(d.reset_index(drop=True))
    m = pd.concat(filas, ignore_index=True)
    m["dir_actual"] = m["direccion"].map({"baja": -1.0, "se_mantiene": 0.0, "sube": 1.0})
    B = ["dif_1", "dif_2", "dif_3", "dir_actual", "es_diesel", "es_regular", "es_super"]
    X = ["fred_1", "fred_2", "fred_3", "wti_1"]
    m = m.dropna(subset=B + X + ["direccion_siguiente"]).copy()
    m["apta"] = True
    return m, B, X

LR_M = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))
metricas_mes = []
for umbral in (0.15, 0.30):
    m, B, X = datos_mensuales(ofi, fred_csv, TIPO_CAMBIO, umbral)
    pr = evaluar(m, {"logit_base": (LR_M(), B), "logit_exo": (LR_M(), B + X)}, 36, 6, usar_col="apta")
    r = resumen(pr, ["repetir", "moda", "logit_base", "logit_exo"], bloque=3).assign(escenario=f"mensual_umbral_{umbral}")
    metricas_mes.append(r)
    print(f"\n=== mensual, umbral Q{umbral} (n meses aptos = {len(m)}) ==="); print(r.round(3).to_string(index=False))
metricas_mes = pd.concat(metricas_mes, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Escribir tablas `_v2` (solo si el widget `escribir` = si)

# COMMAND ----------

from pyspark.sql import functions as F

def guardar(pdf, tabla):
    pdf = pdf.copy()
    for c in pdf.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf[c]):
            pdf[c] = pdf[c].dt.date
        elif pdf[c].dtype == object:
            col = pdf[c].astype(object); col[col.isna()] = None; pdf[c] = col
    sdf = spark.createDataFrame(pdf)
    for c, tipo in sdf.dtypes:
        if tipo in ("double", "float"):
            sdf = sdf.withColumn(c, F.when(F.isnan(c), None).otherwise(F.col(c)))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabla)
    print(f"Escrito {tabla} ({spark.table(tabla).count()} filas)")

if ESCRIBIR:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{ESQUEMA_SALIDA}")
    cols = ["combustible", "semana_inicio", "promedio", "fuente", "familia", "dif_q", "direccion", "direccion_siguiente",
            *FEATURES_BASE, *FEATURES_EXO, "apta_modelo", "apta_exo"]
    guardar(feats[cols], f"{CATALOGO}.{ESQUEMA_SALIDA}.features_modelo_v2")
    guardar(pd.concat([metricas_sem, metricas_mes], ignore_index=True).assign(generado_en=pd.Timestamp.now()),
            f"{CATALOGO}.{ESQUEMA_SALIDA}.metricas_modelo_v2")
else:
    print("escribir = no: no se escribió ninguna tabla (cámbialo a 'si' para crear *_v2)")
