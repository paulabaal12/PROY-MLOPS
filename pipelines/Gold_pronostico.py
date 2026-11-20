# Databricks notebook source
# MAGIC %pip install -q "mlflow>=2.10" "scikit-learn>=1.3"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold: precio semanal, pronóstico de dirección y registro del modelo
# MAGIC
# MAGIC **Orden del pipeline:** `Lector_combinado` → `Ingesta_externas` → `Silver_lecturas` → **`Gold_pronostico`**.
# MAGIC
# MAGIC **Datos:** precio semanal real del MEM (`silver.precios_mem_semanales`, 719 semanas por combustible, sin interpolar) y precio internacional de FRED en Q/galón con el tipo de cambio real (`silver.exogenas_semanales`). No usa MAGA ni las fotos para entrenar.
# MAGIC
# MAGIC **Objetivo:** dirección del precio de la **semana siguiente** (`baja` / `se_mantiene` / `sube`, umbral Q0.10).
# MAGIC
# MAGIC **Modelos:** regresión logística y árbol de gradiente, cada uno con y sin FRED. El **elegido** es el de mayor accuracy promedio en las ventanas de prueba (rolling-origin: se reentrena cada 13 semanas solo con el pasado). Línea base: repetir la dirección anterior. IC95% por bootstrap de bloques de 4 semanas.
# MAGIC
# MAGIC | Tabla | Contenido |
# MAGIC | --- | --- |
# MAGIC | `gold.precio_semanal` | Precio semanal MEM por combustible |
# MAGIC | `gold.variacion_semanal` | Cambio semanal en Q y % y su dirección |
# MAGIC | `gold.features_modelo` | Variables del modelo por semana y combustible |
# MAGIC | `gold.metricas_modelo` | Accuracy, F1 e IC95% por ventana y modelo (`es_elegido`) |
# MAGIC | `gold.predicciones` | Predicción de cada modelo en la ventana de prueba principal |
# MAGIC | `gold.pronostico_proxima_semana` | Pronóstico de la semana siguiente con probabilidades y versión del modelo |
# MAGIC
# MAGIC Modelo registrado en Unity Catalog: `<catálogo>.gold.modelo_direccion_combustible` (alias `champion`). Experimento MLflow: `/Shared/pronostico_combustibles`.

# COMMAND ----------

def _w(n, d, e):
    try:
        dbutils.widgets.text(n, d, e); return dbutils.widgets.get(n)
    except NameError:
        return d
CAT      = _w("catalogo", "equipo4", "Catálogo")
UMBRAL   = float(_w("umbral_direccion", "0.10", "Umbral sube/baja (Q)"))
PRUEBA   = int(_w("semanas_prueba", "104", "Semanas de prueba principal"))
PASO     = int(_w("paso_reentreno", "13", "Reentrenar cada N semanas"))
DESDE    = _w("desde", "2021-03-01", "Primera semana modelable (FRED confiable)")
EXPERIMENTO = _w("experimento_mlflow", "/Shared/pronostico_combustibles", "Experimento MLflow")
MODELO_UC   = f"{CAT}.gold.modelo_direccion_combustible"
ESCRIBIR = _w("escribir", "si", "Escribir tablas (si / no)").lower() == "si"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Datos

# COMMAND ----------

import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
COMB = ["diesel", "regular", "super"]
MAPA = {-1.0: "baja", 0.0: "se_mantiene", 1.0: "sube"}

mem = spark.table(f"{CAT}.silver.precios_mem_semanales").toPandas()
mem["semana_inicio"] = pd.to_datetime(mem["semana_inicio"])
for c in ("promedio", "precio_cierre"): mem[c] = mem[c].astype(float)

ex = spark.table(f"{CAT}.silver.exogenas_semanales").toPandas()
ex["semana_inicio"] = pd.to_datetime(ex["semana_inicio"])
EX = ex.set_index("semana_inicio").sort_index()[["gasolina_gulf_q_gal", "diesel_gulf_q_gal", "wti_usd_barril"]].astype(float)
EX = EX.reindex(pd.date_range(EX.index.min(), max(EX.index.max(), mem.semana_inicio.max()), freq="7D")).ffill()
print(f"MEM: {len(mem)} filas, {mem.semana_inicio.min().date()} a {mem.semana_inicio.max().date()} | FRED: {EX.index.min().date()} a {EX.index.max().date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Features

# COMMAND ----------

BASE = ["dif1", "dif2", "dif3", "dif4", "acum4", "dir", "diesel", "regular", "super"]
EXO  = ["f1", "f2", "f3", "f4", "f6", "f_lag1", "f_lag2", "f_lag3", "wti4"]

def construir(umbral):
    filas = []
    for c in COMB:
        d = (mem[mem.combustible == c].set_index("semana_inicio").sort_index()[["promedio", "precio_cierre", "n_dias"]]
             .rename(columns={"promedio": "p"}).join(EX, how="left"))
        d["dif1"] = d.p.diff()
        for k in (2, 3, 4): d[f"dif{k}"] = d.dif1.shift(k - 1)
        d["acum4"] = d.p - d.p.shift(4)
        d["dir"] = np.select([d.dif1 > umbral, d.dif1 < -umbral], [1.0, -1.0], 0.0)
        d.loc[d.dif1.isna(), "dir"] = np.nan
        g = d.diesel_gulf_q_gal if c == "diesel" else d.gasolina_gulf_q_gal
        for k in (1, 2, 3, 4, 6): d[f"f{k}"] = g.diff(k)
        for k in (1, 2, 3): d[f"f_lag{k}"] = g.diff().shift(k)
        d["wti4"] = d.wti_usd_barril.diff(4)
        d["sig"] = d["dir"].shift(-1)
        d["combustible"] = c
        filas.append(d.rename_axis("semana_inicio").reset_index())
    D = pd.concat(filas, ignore_index=True)
    for c in COMB: D[c] = (D.combustible == c).astype(float)
    return D

D = construir(UMBRAL)
M = D[D.semana_inicio >= DESDE].dropna(subset=BASE + EXO + ["sig"]).copy()
print("filas modelables:", len(M), "| semanas:", M.semana_inicio.nunique(), "| distribución:", M.sig.map(MAPA).value_counts(normalize=True).round(2).to_dict())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluación rolling-origin y elección del modelo

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def LR(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))
def GB(): return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
MODELOS = {"logit_base": (LR, BASE), "logit_fred": (LR, BASE + EXO), "gb_base": (GB, BASE), "gb_fred": (GB, BASE + EXO)}

def rolling(M, semanas_prueba, paso):
    sem = np.sort(M.semana_inicio.unique()); prueba = sem[-semanas_prueba:]; salidas = []
    for i in range(0, len(prueba), paso):
        b = prueba[i:i + paso]; tr = M[M.semana_inicio < b[0]]; te = M[M.semana_inicio.isin(b)]
        o = te[["combustible", "semana_inicio", "sig"]].copy()
        o["repetir"] = te["dir"].to_numpy(); o["moda"] = tr["sig"].mode().iloc[0]
        for n, (mk, cols) in MODELOS.items():
            o[n] = mk().fit(tr[cols], tr["sig"]).predict(te[cols])
        salidas.append(o)
    return pd.concat(salidas, ignore_index=True)

def resumen(R, columnas, base="repetir", n_boot=2000, bloque=4):
    rng = np.random.default_rng(0); real = R["sig"].to_numpy(); sem = np.sort(R.semana_inicio.unique())
    idx = {s: np.where(R.semana_inicio.to_numpy() == s)[0] for s in sem}; filas = []
    for c in columnas:
        a = (R[c].to_numpy() == real).astype(float); b = (R[base].to_numpy() == real).astype(float); difs = []
        for _ in range(n_boot):
            ii = np.concatenate([idx[s] for j in rng.integers(0, max(1, len(sem) - bloque + 1), max(1, len(sem) // bloque)) for s in sem[j:j + bloque]])
            difs.append(a[ii].mean() - b[ii].mean())
        lo, hi = np.percentile(difs, [2.5, 97.5])
        filas.append({"modelo": c, "n": len(R), "semanas": len(sem), "accuracy": a.mean(),
                      "f1_macro": f1_score(real, R[c], average="macro", labels=[-1.0, 0.0, 1.0], zero_division=0),
                      "dif_vs_repetir": a.mean() - b.mean(), "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(filas)

res = []
for nombre, n in (("ultimas_52", 52), (f"ultimas_{PRUEBA}", PRUEBA), ("ultimas_156", 156)):
    R = rolling(M, n, PASO)
    r = resumen(R, ["repetir", "moda"] + list(MODELOS)).assign(escenario=nombre); res.append(r)
    print(f"\n=== {nombre} ==="); print(r.round(3).to_string(index=False))
    if n == PRUEBA: R_prueba = R
metricas = pd.concat(res, ignore_index=True)

prom = metricas[metricas.modelo.isin(MODELOS)].groupby("modelo")["accuracy"].mean()
ELEGIDO = prom.idxmax()
metricas["es_elegido"] = metricas["modelo"] == ELEGIDO
print("\nAccuracy promedio por modelo:", prom.round(3).to_dict(), "| elegido:", ELEGIDO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Modelo final y pronóstico de la próxima semana
# MAGIC El elegido se entrena con todo el histórico etiquetado y predice la dirección de la semana siguiente a la última semana observada.

# COMMAND ----------

mk, cols = MODELOS[ELEGIDO]
final = mk().fit(M[cols], M["sig"])
ult = D[D.semana_inicio == D.semana_inicio.max()].dropna(subset=cols)
proba = final.predict_proba(ult[cols])
pron = ult[["combustible", "semana_inicio", "p"]].rename(columns={"p": "precio_actual"}).copy()
pron["semana_pronosticada"] = pron["semana_inicio"] + pd.Timedelta(days=7)
pron["direccion_pronosticada"] = [MAPA[x] for x in final.predict(ult[cols])]
for i, k in enumerate(final.classes_): pron[f"prob_{MAPA[k]}"] = proba[:, i].round(3)
pron["modelo"] = ELEGIDO
display(pron)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MLflow y Unity Catalog
# MAGIC Una corrida por escenario y modelo, más `modelo_final`, que guarda el modelo entrenado y lo registra en Unity Catalog con el alias `champion`.

# COMMAND ----------

RUN_FINAL = VERSION_UC = None
try:
    import mlflow, mlflow.sklearn
    mlflow.set_experiment(EXPERIMENTO)
    comunes = {"umbral_q": UMBRAL, "desde": DESDE, "paso_reentreno": PASO, "fuente_precio": "mem_semanal", "tipo_cambio": "real_mem"}
    for _, r in metricas.iterrows():
        with mlflow.start_run(run_name=f"{r['escenario']} | {r['modelo']}"):
            mlflow.log_params({**comunes, "escenario": r["escenario"], "modelo": r["modelo"], "usa_fred": "fred" in r["modelo"]})
            mlflow.log_metrics({"accuracy": float(r["accuracy"]), "f1_macro": float(r["f1_macro"]), "dif_vs_repetir": float(r["dif_vs_repetir"]),
                                "ic95_inf": float(r["ic95_inf"]), "ic95_sup": float(r["ic95_sup"]), "n": float(r["n"])})
    with mlflow.start_run(run_name=f"modelo_final | {ELEGIDO}") as run:
        ref = metricas[(metricas.escenario == f"ultimas_{PRUEBA}") & (metricas.modelo == ELEGIDO)].iloc[0]
        mlflow.log_params({**comunes, "modelo": ELEGIDO, "usa_fred": "fred" in ELEGIDO, "entrenado_con": f"{len(M)} filas hasta {M.semana_inicio.max().date()}"})
        mlflow.log_metrics({"accuracy_prueba": float(ref["accuracy"]), "dif_vs_repetir": float(ref["dif_vs_repetir"]), "accuracy_promedio": float(prom[ELEGIDO])})
        mlflow.sklearn.log_model(final, "modelo", input_example=M[cols].head(3))
        RUN_FINAL = run.info.run_id
    print("Experimento:", EXPERIMENTO, "| run final:", RUN_FINAL)
    try:
        mlflow.set_registry_uri("databricks-uc")
        v = mlflow.register_model(f"runs:/{RUN_FINAL}/modelo", MODELO_UC)
        mlflow.MlflowClient().set_registered_model_alias(MODELO_UC, "champion", v.version)
        VERSION_UC = int(v.version)
        print(f"Modelo registrado: {MODELO_UC} v{VERSION_UC} (alias champion)")
    except Exception as e:
        print("No se pudo registrar en Unity Catalog:", str(e)[:300])
except Exception as e:
    print("No se pudo registrar en MLflow:", str(e)[:300])
pron["run_id_mlflow"] = RUN_FINAL
pron["version_modelo_uc"] = VERSION_UC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Escribir tablas gold

# COMMAND ----------

def guardar(pdf, tabla):
    pdf = pdf.copy()
    for c in pdf.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf[c]): pdf[c] = pdf[c].dt.date
        elif pdf[c].dtype == object: pdf[c] = pdf[c].astype("string")
    sdf = spark.createDataFrame(pdf)
    from pyspark.sql import functions as F
    for c, t in sdf.dtypes:
        if t in ("double", "float"): sdf = sdf.withColumn(c, F.when(F.isnan(c), None).otherwise(F.col(c)))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabla)
    print("Escrito", tabla, sdf.count(), "filas")

if ESCRIBIR:
    ahora = pd.Timestamp.now()
    ps = mem[["combustible", "semana_inicio", "promedio", "precio_cierre", "n_dias", "fuente"]].sort_values(["combustible", "semana_inicio"])
    guardar(ps, f"{CAT}.gold.precio_semanal")
    vs = D[["combustible", "semana_inicio", "p", "dif1"]].rename(columns={"p": "promedio", "dif1": "dif_q"}).copy()
    prev = vs.groupby("combustible")["promedio"].shift(1)
    vs["dif_pct"] = (100 * vs["dif_q"] / prev).round(2)
    vs["direccion"] = np.select([vs.dif_q > UMBRAL, vs.dif_q < -UMBRAL], ["sube", "baja"], "se_mantiene")
    vs.loc[vs.dif_q.isna(), "direccion"] = None
    guardar(vs, f"{CAT}.gold.variacion_semanal")
    ft = M[["combustible", "semana_inicio", "p", "dir", "sig"] + BASE[:5] + EXO].rename(columns={"p": "promedio", "dir": "dir_actual", "sig": "dir_siguiente"})
    ft["dir_actual"] = ft["dir_actual"].map(MAPA); ft["dir_siguiente"] = ft["dir_siguiente"].map(MAPA)
    guardar(ft, f"{CAT}.gold.features_modelo")
    guardar(metricas.assign(umbral_q=UMBRAL, generado_en=ahora), f"{CAT}.gold.metricas_modelo")
    p = R_prueba.copy()
    for c in ["sig", "repetir", "moda"] + list(MODELOS): p[c] = p[c].map(MAPA)
    guardar(p.rename(columns={"sig": "real"}), f"{CAT}.gold.predicciones")
    guardar(pron.assign(generado_en=ahora), f"{CAT}.gold.pronostico_proxima_semana")
else:
    print("escribir = no")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Copia de Oro a S3 para la API
# MAGIC
# MAGIC La API del Droplet lee la capa Oro desde S3 (no desde Databricks). Si el parámetro `ruta_oro_s3` trae una ruta
# MAGIC (en el Job: `s3://cc3105-au-869227452861-us-east-2-an/equipo4/gold`), cada tabla Oro se copia como tabla Delta en
# MAGIC `<ruta_oro_s3>/<tabla>` mediante la external location `xloc-au`. Vacío = no se copia nada.

# COMMAND ----------

RUTA_ORO_S3 = _w("ruta_oro_s3", "", "Ruta S3 de Oro para la API (vacío = no copiar)").strip().rstrip("/")
TABLAS_API = ["precio_semanal", "variacion_semanal", "features_modelo",
              "metricas_modelo", "predicciones", "pronostico_proxima_semana"]

if ESCRIBIR and RUTA_ORO_S3:
    for t in TABLAS_API:
        destino = f"{RUTA_ORO_S3}/{t}"
        (spark.table(f"{CAT}.gold.{t}").write.format("delta")
             .mode("overwrite").option("overwriteSchema", "true")
             .option("delta.enableDeletionVectors", "false")  # para que deltalake (la API) la pueda leer
             .save(destino))
        print(f"copia para la API: {destino}")
else:
    print("ruta_oro_s3 vacío o escribir = no: no se copia a S3")
