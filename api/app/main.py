"""API de precios de combustibles en Guatemala (lee solo la capa Oro).

Lee las tablas Delta que escribe el notebook Gold_pronostico desde GOLD_URI:
  - en el servidor:  s3://<bucket-oro>/equipo-04/gold   (credenciales de solo lectura en .env)
  - en tu compu:     /app/datos_ejemplo                 (datos simulados con el mismo esquema)

Tablas que usa:
  precio_semanal             combustible, semana_inicio, promedio, precio_cierre, n_dias, fuente
  pronostico_proxima_semana  combustible, semana_inicio, precio_actual, semana_pronosticada, direccion_pronosticada,
                             prob_baja, prob_se_mantiene, prob_sube, modelo, run_id_mlflow, version_modelo_uc, generado_en
  metricas_modelo            modelo, escenario, accuracy, f1_macro, dif_vs_repetir, ic95_inf, ic95_sup, es_elegido, ...
  predicciones               combustible, semana_inicio, real, repetir, moda, <una columna por modelo>
"""
import math
import os
import time
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import pandas as pd
from deltalake import DeltaTable
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

GOLD_URI = os.getenv("GOLD_URI", "/app/datos_ejemplo").rstrip("/")
CACHE_SEG = int(os.getenv("CACHE_SEGUNDOS", "300"))            # cada cuánto se vuelve a leer S3
ESCENARIO = os.getenv("ESCENARIO_METRICAS", "ultimas_104")      # ventana de prueba principal del notebook Gold
LINEA_BASE = "repetir"                                           # "se repite la dirección de la semana anterior"
TABLAS = ["precio_semanal", "pronostico_proxima_semana", "metricas_modelo", "predicciones"]
COLUMNAS_FECHA = {"semana_inicio", "semana_pronosticada", "generado_en"}
CLASES = ["baja", "se_mantiene", "sube"]

app = FastAPI(
    title="Precios de combustibles Guatemala — Equipo 04",
    description="Precio semanal (martes a lunes) de súper, regular y diésel, y predicción de si sube, baja o se "
                "mantiene la semana siguiente. Datos de la capa Oro del pipeline en Databricks.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


class Combustible(str, Enum):
    diesel = "diesel"
    regular = "regular"
    super = "super"


# ---------- lectura de la capa Oro (con caché en memoria) ----------
_cache: dict = {}


def _opciones_s3() -> Optional[dict]:
    if not GOLD_URI.startswith("s3://"):
        return None
    opciones = {"AWS_REGION": os.getenv("AWS_REGION", "us-east-2")}
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        if os.getenv(k):
            opciones[k] = os.environ[k]
    return opciones


def leer(tabla: str) -> pd.DataFrame:
    ahora = time.time()
    if tabla in _cache and ahora - _cache[tabla][0] < CACHE_SEG:
        return _cache[tabla][1]
    try:
        df = DeltaTable(f"{GOLD_URI}/{tabla}", storage_options=_opciones_s3()).to_pandas()
    except Exception as e:
        if tabla in _cache:                      # si S3 falla, se sirve lo último que se leyó
            return _cache[tabla][1]
        raise HTTPException(503, f"No pude leer la tabla Oro '{tabla}': {type(e).__name__}: {e}")
    for c in df.columns:
        if c in COLUMNAS_FECHA:
            df[c] = pd.to_datetime(df[c]).dt.date
    _cache[tabla] = (ahora, df)
    return df


def _valor(v):
    """Valor de pandas/numpy -> JSON (fechas como texto, NaN/NA como null)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):                       # numpy int/float/bool
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def registros(df: pd.DataFrame) -> list[dict]:
    return [{k: _valor(v) for k, v in fila.items()} for fila in df.to_dict(orient="records")]


def _con_columnas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Selecciona columnas; las que no existan en la tabla salen como null (por si cambia el esquema)."""
    return df.reindex(columns=columnas)


def _precio_semanal() -> pd.DataFrame:
    """precio_semanal + semana_fin y el cambio contra la semana anterior (umbral del modelo)."""
    df = leer("precio_semanal").sort_values(["combustible", "semana_inicio"]).copy()
    df["semana_fin"] = df["semana_inicio"].map(lambda d: d + timedelta(days=6))
    anterior = df.groupby("combustible")["promedio"].shift(1)
    df["dif_q"] = (df["promedio"] - anterior).round(2)
    df["dif_pct"] = (100 * df["dif_q"] / anterior).round(2)
    umbral = _umbral()
    df["direccion"] = df["dif_q"].map(
        lambda d: None if pd.isna(d) else "sube" if d > umbral else "baja" if d < -umbral else "se_mantiene")
    return df


def _umbral() -> float:
    try:
        m = leer("metricas_modelo")
        if "umbral_q" in m.columns and m["umbral_q"].notna().any():
            return float(m["umbral_q"].dropna().iloc[0])
    except HTTPException:
        pass
    return 0.10


COLS_PRECIO = ["combustible", "semana_inicio", "semana_fin", "promedio", "precio_cierre", "n_dias",
               "dif_q", "dif_pct", "direccion", "fuente"]


# ---------- endpoints ----------
@app.get("/health", summary="Estado del servicio y de cada tabla Oro")
def health():
    estado = {}
    for t in TABLAS:
        try:
            estado[t] = f"ok ({len(leer(t))} filas)"
        except HTTPException as e:
            estado[t] = f"error: {e.detail}"
    todo_ok = all(v.startswith("ok") for v in estado.values())
    ultima = None
    if estado["precio_semanal"].startswith("ok"):
        ultima = _valor(leer("precio_semanal")["semana_inicio"].max())
    return {"status": "ok" if todo_ok else "degradado", "origen_datos": GOLD_URI,
            "ultima_semana_con_precio": ultima, "tablas": estado}


@app.get("/precios/actual", summary="Último precio semanal por combustible")
def precios_actual():
    df = _precio_semanal()
    ultimo = df.groupby("combustible").tail(1).sort_values("combustible")
    return registros(_con_columnas(ultimo, COLS_PRECIO))


@app.get("/precios", summary="Histórico semanal con filtros y paginación")
def precios(
    combustible: Optional[Combustible] = Query(None, description="diesel, regular o super; vacío = todos"),
    desde: Optional[date] = Query(None, description="Fecha inicial AAAA-MM-DD (incluye la semana que la contiene)"),
    hasta: Optional[date] = Query(None, description="Fecha final AAAA-MM-DD"),
    limit: int = Query(100, ge=1, le=1000, description="Máximo de filas"),
    offset: int = Query(0, ge=0, description="Filas a saltar (paginación)"),
):
    if desde and hasta and desde > hasta:
        raise HTTPException(400, "'desde' no puede ser mayor que 'hasta'")
    df = _precio_semanal()
    if combustible:
        df = df[df["combustible"] == combustible.value]
    if desde:
        df = df[df["semana_fin"] >= desde]          # semanas que tocan el rango
    if hasta:
        df = df[df["semana_inicio"] <= hasta]
    df = df.sort_values(["semana_inicio", "combustible"], ascending=[False, True])
    pagina = df.iloc[offset:offset + limit]
    return {"total": len(df), "limit": limit, "offset": offset,
            "datos": registros(_con_columnas(pagina, COLS_PRECIO))}


def _fila_prediccion(f: dict) -> dict:
    probs = {c: _valor(f.get(f"prob_{c}")) for c in CLASES}
    prediccion = f.get("direccion_pronosticada")
    return {
        "combustible": f.get("combustible"),
        "semana": _valor(f.get("semana_pronosticada")),
        "prediccion": prediccion,
        "probabilidad": probs.get(prediccion),
        "prob_baja": probs["baja"], "prob_se_mantiene": probs["se_mantiene"], "prob_sube": probs["sube"],
        "semana_base": _valor(f.get("semana_inicio")),
        "precio_actual": _valor(f.get("precio_actual")),
        "modelo": f.get("modelo"),
        "version_modelo_uc": _valor(f.get("version_modelo_uc")),
        "generado_en": _valor(f.get("generado_en")),
    }


@app.get("/prediccion", summary="Predicción de la semana siguiente")
def prediccion(combustible: Optional[Combustible] = Query(None, description="diesel, regular o super; vacío = todos")):
    df = leer("pronostico_proxima_semana")
    if combustible:
        df = df[df["combustible"] == combustible.value]
    if df.empty:
        raise HTTPException(404, "No hay predicción para ese combustible")
    salida = [_fila_prediccion(f) for f in df.sort_values("combustible").to_dict(orient="records")]
    return salida[0] if combustible else salida


def _metricas_escenario(escenario: Optional[str]):
    df = leer("metricas_modelo")
    disponibles = sorted(df["escenario"].dropna().unique().tolist()) if "escenario" in df.columns else []
    if escenario is None:
        escenario = ESCENARIO if ESCENARIO in disponibles else (disponibles[len(disponibles) // 2] if disponibles else None)
    elif escenario not in disponibles:
        raise HTTPException(404, f"Escenario '{escenario}' no existe. Disponibles: {disponibles}")
    if escenario is not None:
        df = df[df["escenario"] == escenario]
    return df, escenario, disponibles


def _modelo_elegido() -> Optional[str]:
    m = leer("metricas_modelo")
    elegidos = m.loc[m["es_elegido"].fillna(False).astype(bool), "modelo"].dropna().unique()
    return str(elegidos[0]) if len(elegidos) else None


@app.get("/modelo/metricas", summary="Exactitud del modelo y de la línea base")
def modelo_metricas(escenario: Optional[str] = Query(
        None, description="Ventana de prueba, ej. ultimas_52, ultimas_104, ultimas_156; vacío = la principal")):
    df, escenario, disponibles = _metricas_escenario(escenario)
    nombre = _modelo_elegido()
    elegido = df[df["modelo"] == nombre]
    base = df[df["modelo"] == LINEA_BASE]
    acc_m = float(elegido["accuracy"].iloc[0]) if len(elegido) else None
    acc_b = float(base["accuracy"].iloc[0]) if len(base) else None
    fila = elegido.iloc[0].to_dict() if len(elegido) else {}
    return {
        "modelo_elegido": nombre,
        "escenario": escenario,
        "escenarios_disponibles": disponibles,
        "validacion": "temporal (rolling-origin): cada bloque de semanas se predice con un modelo entrenado solo con el pasado",
        "linea_base": "repetir: se repite la dirección de la semana anterior",
        "exactitud_modelo": acc_m,
        "exactitud_linea_base": acc_b,
        "supera_linea_base": (acc_m > acc_b) if acc_m is not None and acc_b is not None else None,
        "diferencia_vs_linea_base": _valor(fila.get("dif_vs_repetir")),
        "ic95_diferencia": [_valor(fila.get("ic95_inf")), _valor(fila.get("ic95_sup"))] if fila else None,
        "f1_macro_modelo": _valor(fila.get("f1_macro")),
        "semanas_evaluadas": _valor(fila.get("semanas")),
        "detalle": registros(df.sort_values("accuracy", ascending=False)),
    }


@app.get("/prediccion/historial", summary="Predicciones pasadas del modelo elegido contra la dirección real")
def prediccion_historial(
    combustible: Optional[Combustible] = Query(None, description="diesel, regular o super; vacío = todos"),
    limit: int = Query(52, ge=1, le=1000, description="Máximo de filas (las más recientes)"),
    offset: int = Query(0, ge=0, description="Filas a saltar (paginación)"),
):
    df = leer("predicciones").copy()
    nombre = _modelo_elegido()
    if nombre not in df.columns:
        raise HTTPException(503, f"La tabla predicciones no tiene la columna del modelo elegido ('{nombre}')")
    if combustible:
        df = df[df["combustible"] == combustible.value]
    df["semana"] = df["semana_inicio"].map(lambda d: d + timedelta(days=7))
    df = df.rename(columns={nombre: "prediccion", LINEA_BASE: "prediccion_linea_base", "real": "direccion_real"})
    df["acierto"] = df["prediccion"] == df["direccion_real"]
    df["acierto_linea_base"] = df["prediccion_linea_base"] == df["direccion_real"]
    df = df.sort_values(["semana", "combustible"], ascending=[False, True])
    pagina = df.iloc[offset:offset + limit]
    return {
        "modelo": nombre, "total": len(df), "limit": limit, "offset": offset,
        "exactitud_modelo": round(float(df["acierto"].mean()), 4) if len(df) else None,
        "exactitud_linea_base": round(float(df["acierto_linea_base"].mean()), 4) if len(df) else None,
        "datos": registros(pagina[["combustible", "semana", "prediccion", "prediccion_linea_base", "direccion_real",
                                   "acierto"]]),
    }


@app.get("/", include_in_schema=False)
def inicio():
    return {"mensaje": "API de precios de combustibles — Equipo 04", "documentacion": "/docs",
            "endpoints": ["/health", "/precios/actual", "/precios", "/prediccion", "/prediccion/historial",
                          "/modelo/metricas"]}
