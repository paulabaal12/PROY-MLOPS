# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: lecturas de fotos validadas contra el MEM diario
# MAGIC
# MAGIC **Orden del pipeline:** `Lector_combinado` → `Ingesta_externas` → **`Silver_lecturas`** → `Gold_pronostico`.
# MAGIC
# MAGIC Valida cada lectura contra el **precio MEM del día de la foto** (no usa MAGA ni versiones anteriores de silver), así las fotos de 2026 también se validan.
# MAGIC
# MAGIC | Entrada | Uso |
# MAGIC | --- | --- |
# MAGIC | `bronze.lecturas_raw` | Salida de `Lector_combinado`, una fila por foto |
# MAGIC | `bronze.precios_mem_diarios` | Precio de referencia del día (el último dato disponible, hasta 7 días antes) |
# MAGIC | `silver/archivos/correcciones*.csv` | Correcciones manuales (`archivo,columna,valor`); en duplicados gana la última |
# MAGIC
# MAGIC **Regla por combustible (diésel, regular, súper):** corrección manual → lectura de la foto si está en rango y a `tolerancia` Q del MEM → precio MEM del día.
# MAGIC **V-Power** no existe en el MEM: se acepta la corrección manual o una lectura que esté entre Q0 y Q2.5 sobre el súper final.
# MAGIC
# MAGIC | Salida | Contenido |
# MAGIC | --- | --- |
# MAGIC | `silver.lecturas_fotos` | Una fila por foto, con precio final, origen (`corregido_manual`, `foto`, `oficial_mem`), lo leído y el motivo del descarte |
# MAGIC | `silver.revision_fotos` | Fotos con algún precio no confiable, con sugerencia si la lectura coincide con otro combustible |
# MAGIC | `silver.validacion_fotos_vs_mem` | Lectura original contra MEM del mismo día, por origen final |

# COMMAND ----------

def _w(n, d, e):
    try:
        dbutils.widgets.text(n, d, e); return dbutils.widgets.get(n)
    except NameError:
        return d
CAT        = _w("catalogo", "equipo4", "Catálogo")
TOL        = float(_w("tolerancia_q", "1.0", "Tolerancia lectura vs MEM (Q)"))
CORR_FILES = [x.strip() for x in _w("correcciones", "correcciones.csv", "Archivos de correcciones (coma)").split(",") if x.strip()]
RANGO      = (20.0, 50.0)

# COMMAND ----------

import numpy as np, pandas as pd, os
COMB, TODOS = ["diesel", "regular", "super"], ["diesel", "regular", "super", "v_power"]

lect = spark.table(f"{CAT}.bronze.lecturas_raw").toPandas()
lect["procesado_en"] = pd.to_datetime(lect["procesado_en"])
lect = lect.sort_values("procesado_en").drop_duplicates("archivo", keep="last").copy()
lect["fecha_hora"] = pd.to_datetime(lect["fecha_hora"])
if lect["fecha_hora"].dt.tz is not None:
    lect["fecha_hora"] = lect["fecha_hora"].dt.tz_convert(None)
lect["fecha"] = lect["fecha_hora"].dt.normalize()
print(len(lect), "fotos | sin fecha:", int(lect["fecha"].isna().sum()))

# --- MEM del día ---
mem = spark.table(f"{CAT}.bronze.precios_mem_diarios").select("fecha", *COMB).toPandas()
mem["fecha"] = pd.to_datetime(mem["fecha"]); mem = mem.sort_values("fecha").rename(columns={c: f"mem_{c}" for c in COMB})
lect = lect.sort_values("fecha")
con_fecha = lect["fecha"].notna()
lect = pd.concat([pd.merge_asof(lect[con_fecha], mem, on="fecha", direction="backward", tolerance=pd.Timedelta(days=7)), lect[~con_fecha]], ignore_index=True)
print("fotos sin referencia MEM:", int(lect["mem_diesel"].isna().sum()))

# --- Correcciones manuales ---
partes = []
for f in CORR_FILES:
    ruta = f"/Volumes/{CAT}/silver/archivos/{f}"
    if os.path.exists(ruta):
        d = pd.read_csv(ruta); d["_archivo_csv"] = f; partes.append(d)
    else:
        print("no existe", ruta)
corr = pd.concat(partes, ignore_index=True)
corr["archivo"] = corr["archivo"].astype(str).str.strip(); corr["columna"] = corr["columna"].astype(str).str.lower().str.strip()
corr["valor"] = pd.to_numeric(corr["valor"], errors="coerce")
corr = corr[corr["columna"].isin(TODOS) & corr["valor"].notna()].drop_duplicates(["archivo", "columna"], keep="last")
print("correcciones:", len(corr), "en", corr["archivo"].nunique(), "fotos | no encontradas en bronze:", sorted(set(corr["archivo"]) - set(lect["archivo"])))
cw = corr.pivot(index="archivo", columns="columna", values="valor").add_prefix("corr_")
lect = lect.merge(cw, left_on="archivo", right_index=True, how="left")
for c in TODOS:
    if f"corr_{c}" not in lect.columns: lect[f"corr_{c}"] = np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limpieza por combustible

# COMMAND ----------

def pista(fila, c, leido):
    """Si la lectura descartada coincide con el MEM de otro combustible, lo sugiere."""
    otros = [k for k in COMB if k != c and pd.notna(fila[f"mem_{k}"]) and abs(leido - fila[f"mem_{k}"]) <= TOL]
    return f" ¿es {otros[0]}?" if otros else ""

for c in COMB:
    l = pd.to_numeric(lect[c], errors="coerce"); corr_c, mem_c = lect[f"corr_{c}"], lect[f"mem_{c}"]
    ok = l.between(*RANGO); cerca = (l - mem_c).abs() <= TOL
    lect[f"{c}_leido"] = l
    lect[c] = np.select([corr_c.notna(), ok & mem_c.notna() & cerca, ok & mem_c.isna()], [corr_c, l, l], default=mem_c).round(2)
    lect[f"{c}_origen"] = [("corregido_manual" if pd.notna(cc) else "foto" if (o and pd.notna(m) and ce) else "foto_sin_referencia" if (o and pd.isna(m))
                            else "oficial_mem" if pd.notna(m) else None) for cc, o, m, ce in zip(corr_c, ok, mem_c, cerca)]
    motivo = []
    for i, f in lect.iterrows():
        li = f[f"{c}_leido"]
        if pd.notna(f[f"corr_{c}"]): motivo.append(None)
        elif pd.isna(li): motivo.append("no se leyó")
        elif not (RANGO[0] <= li <= RANGO[1]): motivo.append("fuera de rango")
        elif pd.notna(f[f"mem_{c}"]) and abs(li - f[f"mem_{c}"]) > TOL:
            motivo.append(f"lejos del MEM ({li - f[f'mem_{c}']:+.2f})" + pista(f, c, li))
        else: motivo.append(None)
    lect[f"{c}_descartado"] = motivo

# V-Power: sin referencia MEM; debe quedar entre Q0 y Q2.5 sobre el súper final
lect["v_power_leido"] = pd.to_numeric(lect["v_power"], errors="coerce")
def _vp(f):
    li, cr = f["v_power_leido"], f["corr_v_power"]
    if pd.notna(cr): return cr, "corregido_manual", None
    if pd.isna(li): return np.nan, None, "no se leyó"
    if not RANGO[0] <= li <= RANGO[1]: return np.nan, None, "fuera de rango"
    if not 0 <= li - f["super"] <= 2.5: return np.nan, None, "no cuadra con el súper"
    return round(li, 2), "foto_vs_super", None
_r = lect.apply(_vp, axis=1, result_type="expand")
lect["v_power"], lect["v_power_origen"], lect["v_power_descartado"] = _r[0].astype(float), _r[1], _r[2]

# La pantalla de venta confirma qué combustible se compró
def coincide(f):
    if pd.isna(f["precio_implicito"]): return None
    for c in TODOS:
        if pd.notna(f[c]) and abs(f[c] - f["precio_implicito"]) <= 0.05: return c
    return None
lect["venta_coincide"] = lect.apply(coincide, axis=1)

orig = lect[[f"{c}_origen" for c in TODOS]]
lect["precios_de_foto"]    = orig.isin(["foto", "foto_sin_referencia", "foto_vs_super"]).sum(axis=1)
lect["precios_corregidos"] = (orig == "corregido_manual").sum(axis=1)
lect["precios_rellenados"] = (orig == "oficial_mem").sum(axis=1)
lect["precios_vacios"]     = orig.isna().sum(axis=1)
lect["procesado_en"] = pd.Timestamp.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardar

# COMMAND ----------

def guardar(pdf, tabla):
    pdf = pdf.copy()
    for c in pdf.columns:
        if pdf[c].dtype == object: pdf[c] = pdf[c].astype("string")
    sdf = spark.createDataFrame(pdf)
    from pyspark.sql import functions as F
    for c, t in sdf.dtypes:
        if t in ("double", "float"): sdf = sdf.withColumn(c, F.when(F.isnan(c), None).otherwise(F.col(c)))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tabla)
    print("Escrito", tabla, sdf.count(), "filas")

lect["anio"] = lect["fecha_hora"].dt.year; lect["mes"] = lect["fecha_hora"].dt.month
cols = (["archivo", "carpeta", "fecha_hora", "fecha", "anio", "mes", "dispositivo", "latitud", "longitud", "altitud_m"]
        + [x for c in TODOS for x in (c, f"{c}_origen", f"{c}_leido", f"{c}_descartado")]
        + [f"mem_{c}" for c in COMB]
        + ["confianza", "venta_estado", "venta_total_q", "venta_galones", "precio_implicito", "venta_coincide",
           "precios_de_foto", "precios_corregidos", "precios_rellenados", "precios_vacios", "procesado_en"])
silver = lect[[c for c in cols if c in lect.columns]].sort_values("fecha_hora")
silver["fecha"] = silver["fecha"].dt.date
guardar(silver, f"{CAT}.silver.lecturas_fotos")

# --- Revisión: fotos con algún precio no confiable ---
hay = silver[[f"{c}_descartado" for c in COMB]].notna().any(axis=1) | silver["v_power_descartado"].notna() & (silver["v_power_descartado"] != "no se leyó")
rev = silver[hay].copy()
rev["motivos"] = rev.apply(lambda f: "; ".join(f"{c}: {f[c + '_descartado']}" for c in TODOS if pd.notna(f[c + "_descartado"]) and f[c + "_descartado"] != "no se leyó" or (c in COMB and pd.notna(f[c + "_descartado"]))), axis=1)
rev = rev[["archivo", "fecha", "motivos", "diesel_leido", "regular_leido", "super_leido", "v_power_leido", "mem_diesel", "mem_regular", "mem_super", "precio_implicito", "venta_coincide"]]
guardar(rev, f"{CAT}.silver.revision_fotos")

# --- Validación: lectura original contra MEM del mismo día ---
filas = []
for c in COMB:
    d = silver[["archivo", "fecha", f"{c}_origen", f"{c}_leido", f"mem_{c}"]].copy(); d.columns = ["archivo", "fecha", "origen", "precio_foto", "precio_mem"]
    d["combustible"] = c; filas.append(d)
val = pd.concat(filas); val = val[val["precio_foto"].notna() & val["precio_mem"].notna()].copy()
val["dif"] = (val["precio_foto"] - val["precio_mem"]).round(2)
guardar(val[["archivo", "fecha", "combustible", "origen", "precio_foto", "precio_mem", "dif"]], f"{CAT}.silver.validacion_fotos_vs_mem")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumen

# COMMAND ----------

res = pd.concat([silver[[f"{c}_origen"]].rename(columns={f"{c}_origen": "origen"}).assign(combustible=c) for c in COMB]).fillna({"origen": "vacío"})
print(res.groupby(["combustible", "origen"]).size().unstack(fill_value=0))
print("fotos para revisar:", len(rev), "| v_power con valor:", int(silver["v_power"].notna().sum()), "| ventas con precio:", int(silver["precio_implicito"].notna().sum()))
