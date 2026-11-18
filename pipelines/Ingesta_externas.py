# Databricks notebook source
# MAGIC %md
# MAGIC # Ingesta de datos externos: MEM (precio real) y FRED (precio internacional)
# MAGIC
# MAGIC **Orden del pipeline:** `Lector_combinado` → **`Ingesta_externas`** → `Silver_lecturas` → `Gold_pronostico`.
# MAGIC
# MAGIC Fuente: Dirección General de Hidrocarburos (MEM), *Precios promedio monitoreados en ciudad capital*, diario desde 2013-02-08.
# MAGIC Es el precio real **semana a semana**; MAGA solo es mensual y termina en 2025-12.
# MAGIC
# MAGIC | Capa | Tabla | Contenido |
# MAGIC | --- | --- | --- |
# MAGIC | Bronce | `bronze.precios_mem_diarios` | Un día por fila: tipo de cambio, súper, regular y diésel (Q/galón) |
# MAGIC | Plata | `silver.precios_mem_semanales` | Semanas martes a lunes (como el MEM): promedio, cierre y días con dato |
# MAGIC | Bronce | `bronze/archivos/fred/*.csv` | CSV de FRED tal cual se descargan |
# MAGIC | Plata | `silver.exogenas_diarias` | Gasolina y diésel de la Costa del Golfo (USD/galón) y WTI (USD/barril), por día |
# MAGIC | Plata | `silver.exogenas_semanales` | Lo mismo por semana martes a lunes, con el tipo de cambio real del MEM y en Q/galón |
# MAGIC # MAGIC
# MAGIC MAGA ya no se usa: el MEM cubre todo el periodo. Para actualizar: sube el nuevo Excel a `/Volumes/equipo4/bronze/archivos/mem/` y cambia el widget `archivo`.

# COMMAND ----------

# MAGIC %pip install -q openpyxl

# COMMAND ----------

dbutils.widgets.text("catalogo", "equipo4", "Catálogo")
dbutils.widgets.text("archivo", "Precios-Promedio-Nacionales-Diarios-2026.xlsx", "Excel en el volumen mem/")
CAT = dbutils.widgets.get("catalogo")
RUTA = f"/Volumes/{CAT}/bronze/archivos/mem/" + dbutils.widgets.get("archivo")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronce: leer el Excel y limpiar
# MAGIC El encabezado está en la fila 7 y la fila 8 trae las unidades. Algunas celdas traen espacios no separables (`\xa0`), por eso se convierten a número explícitamente.

# COMMAND ----------

import pandas as pd
raw = pd.read_excel(RUTA, header=6, skiprows=[7]).iloc[:, :5]
raw.columns = ["fecha", "tipo_cambio", "super", "regular", "diesel"]
raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce")
raw = raw.dropna(subset=["fecha"]).copy()
for c in ["tipo_cambio", "super", "regular", "diesel"]:
    raw[c] = pd.to_numeric(raw[c].astype(str).str.replace("\xa0", "", regex=False).str.strip(), errors="coerce")
raw["fecha"] = raw["fecha"].dt.date
# El Excel repite 3 fechas con valores idénticos (2015-08-25, 2017-05-01, 2017-05-02): se deja una sola fila por fecha.
dup = raw[raw["fecha"].duplicated(keep=False)]
assert dup.groupby("fecha").nunique(dropna=False).to_numpy().max() == 1, "fechas repetidas con valores distintos"
raw = raw.drop_duplicates(subset="fecha").sort_values("fecha")
print(len(raw), "días:", raw["fecha"].min(), "a", raw["fecha"].max(), "| nulos:", raw.isna().sum().to_dict())

(spark.createDataFrame(raw).selectExpr("fecha", "cast(tipo_cambio as double) tipo_cambio", "cast(super as double) super",
                                       "cast(regular as double) regular", "cast(diesel as double) diesel", "current_timestamp() cargado_en")
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CAT}.bronze.precios_mem_diarios"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Plata: semanas martes a lunes
# MAGIC `semana_inicio` es el martes. `dayofweek` va de 1 (domingo) a 7 (sábado), por eso el martes es 3.

# COMMAND ----------

spark.sql(f"""
create or replace table {CAT}.silver.precios_mem_semanales as
with u as (select fecha, stack(3,'diesel',diesel,'regular',regular,'super',super) as (combustible,precio)
           from {CAT}.bronze.precios_mem_diarios),
s as (select combustible, fecha, precio, date_sub(fecha, pmod(dayofweek(fecha)-3,7)) semana_inicio from u where precio is not null)
select combustible, semana_inicio, round(avg(precio),2) promedio, max_by(precio,fecha) precio_cierre, count(*) n_dias, 'mem' fuente
from s group by 1,2""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. FRED: precio internacional (variable externa del modelo)
# MAGIC Descarga las 3 series desde `desde`. Si FRED no responde, se usan los CSV que ya estén en el volumen (el notebook no se cae).

# COMMAND ----------

import os, requests, datetime as dt
try:
    dbutils.widgets.text("fred_desde", "2013-01-01", "FRED desde (AAAA-MM-DD)")
    DESDE = dbutils.widgets.get("fred_desde")
except NameError:
    DESDE = "2013-01-01"
SERIES = {"DGASUSGULF": "gasolina_gulf_usd_gal", "DDFUELUSGULF": "diesel_gulf_usd_gal", "DCOILWTICO": "wti_usd_barril"}
CARPETA = f"/Volumes/{CAT}/bronze/archivos/fred"
os.makedirs(CARPETA, exist_ok=True)
HOY = dt.date.today().isoformat()
for serie in SERIES:
    try:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}&cosd={DESDE}", timeout=45, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if not r.text.lower().startswith(("observation_date", "date")): raise ValueError("la respuesta no parece un CSV de FRED")
        with open(f"{CARPETA}/{serie}_{HOY}.csv", "w") as f: f.write(r.text)
        print(f"OK {serie}: {len(r.text.splitlines()) - 1} filas desde {DESDE}")
    except Exception as e:
        print(f"FRED no respondió para {serie} ({str(e)[:80]}). Se usa el último CSV guardado en {CARPETA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Plata: series diarias y semanales de FRED
# MAGIC Se usa el CSV más reciente y completo (el de más filas) de cada serie. En la semanal, gasolina y diésel se pasan a Q/galón con el **tipo de cambio real promedio de esa semana (MEM)**.

# COMMAND ----------

filas = []
for serie in SERIES:
    cands = sorted(a for a in os.listdir(CARPETA) if a.startswith(serie) and a.endswith(".csv"))
    if not cands: raise RuntimeError(f"No hay CSV de {serie} en {CARPETA}")
    mejor = max(cands, key=lambda a: (len(open(f"{CARPETA}/{a}").read().splitlines()), a))
    d = pd.read_csv(f"{CARPETA}/{mejor}"); d.columns = ["fecha", "valor"]
    d["fecha"] = pd.to_datetime(d["fecha"]).dt.date; d["valor"] = pd.to_numeric(d["valor"], errors="coerce")
    d = d.dropna(subset=["valor"]).assign(serie=serie, fuente_csv=mejor)
    filas.append(d); print(f"{serie}: {len(d)} días ({d['fecha'].min()} a {d['fecha'].max()}) desde {mejor}")
diarias = pd.concat(filas, ignore_index=True)
assert not diarias.duplicated(["fecha", "serie"]).any() and (diarias["valor"] > 0).all()
(spark.createDataFrame(diarias[["fecha", "serie", "valor", "fuente_csv"]]).selectExpr("*", "current_timestamp() procesado_en")
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CAT}.silver.exogenas_diarias"))

def martes(f):
    f = pd.to_datetime(f).dt.normalize()
    return f - pd.to_timedelta((f.dt.dayofweek - 1) % 7, unit="D")
d = diarias.copy(); d["semana_inicio"] = martes(d["fecha"])
sem = d.pivot_table(index="semana_inicio", columns="serie", values="valor", aggfunc="mean").rename(columns=SERIES)
tc = raw.assign(semana_inicio=martes(pd.to_datetime(raw["fecha"]))).groupby("semana_inicio")["tipo_cambio"].mean().rename("tipo_cambio")
sem = sem.join(tc, how="left"); sem["tipo_cambio"] = sem["tipo_cambio"].ffill().bfill()
sem["gasolina_gulf_q_gal"] = (sem["gasolina_gulf_usd_gal"] * sem["tipo_cambio"]).round(3)
sem["diesel_gulf_q_gal"] = (sem["diesel_gulf_usd_gal"] * sem["tipo_cambio"]).round(3)
sem = sem.round({"tipo_cambio": 4}).reset_index(); sem["semana_inicio"] = sem["semana_inicio"].dt.date
(spark.createDataFrame(sem).selectExpr("*", "current_timestamp() procesado_en")
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CAT}.silver.exogenas_semanales"))
print(f"silver.exogenas_semanales: {len(sem)} semanas ({sem['semana_inicio'].min()} a {sem['semana_inicio'].max()})")
