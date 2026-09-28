"""Servicio de predicción: carga el artefacto una vez al arrancar y expone la ficha y las predicciones.

    uvicorn src.app.main:app --reload --port 8000     ->  http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.app.esquemas import Cancion, Prediccion, Salud
from src.artefacto import cargar

RUTA_MODELO = os.environ.get("RUTA_MODELO", "models/modelo.joblib")
RUTA_PUBLIC = os.path.join(os.path.dirname(__file__), "..", "..", "public")

ARTEFACTO = None


@asynccontextmanager
async def ciclo_de_vida(app):
    """Carga el modelo una sola vez, al arrancar el servicio."""
    global ARTEFACTO
    if os.path.exists(RUTA_MODELO):
        ARTEFACTO = cargar(RUTA_MODELO)
        f = ARTEFACTO["ficha"]
        print(f"Modelo cargado: {f['nombre']} ({f['tarea']}, {f.get('fecha')})")
    else:
        print(f"No existe {RUTA_MODELO}. Ejecute: python -m src.entrenar")
    yield


app = FastAPI(
    title="Predicción de popularidad de canciones",
    description="Data Analytics, Semana 9. El modelo es un insumo; la decisión es el producto.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def etiqueta(ficha):
    return f"{ficha['nombre']} ({ficha.get('fecha', '')})"


def exigir_modelo():
    if ARTEFACTO is None:
        raise HTTPException(status_code=503, detail=f"No hay modelo cargado en {RUTA_MODELO}. "
                                                    "Ejecute python -m src.entrenar y reinicie.")
    return ARTEFACTO


@app.get("/", include_in_schema=False)
def raiz():
    """Cliente web (public/index.html). La documentación interactiva sigue en /docs."""
    return FileResponse(os.path.join(RUTA_PUBLIC, "index.html"))


app.mount("/public", StaticFiles(directory=RUTA_PUBLIC), name="public")


@app.get("/salud", response_model=Salud)
def salud():
    ok = ARTEFACTO is not None
    return Salud(estado="ok" if ok else "sin_modelo", modelo_cargado=ok, ruta_modelo=RUTA_MODELO,
                 modelo=etiqueta(ARTEFACTO["ficha"]) if ok else None)


@app.get("/modelo")
def modelo():
    """La ficha del modelo: qué predice, con qué datos, qué tan bien y qué decisión sostiene."""
    return exigir_modelo()["ficha"]


def predecir_lote(canciones: List[Cancion]) -> List[Prediccion]:
    art = exigir_modelo()
    ficha, pipeline = art["ficha"], art["pipeline"]
    X = pd.DataFrame([c.model_dump() for c in canciones])[art["columnas_entrada"]]
    nombre = etiqueta(ficha)
    salida = []
    if ficha["tarea"] == "regresion":
        mae = float(ficha["desempeno"]["mae"])
        corte = float(ficha["decision"]["corte_promocion"])
        for p in pipeline.predict(X):
            p = float(p)
            salida.append(Prediccion(
                tarea="regresion", modelo=nombre, popularidad_esperada=round(p, 1),
                rango=[round(p - mae, 1), round(p + mae, 1)], corte_promocion=corte,
                promocionar=bool(p >= corte),
                explicacion=f"popularidad esperada {p:.1f} (±{mae}) {'supera' if p >= corte else 'no supera'} el corte {corte:g}"))
    else:
        umbral = float(ficha["decision"]["umbral"])
        for pr in pipeline.predict_proba(X)[:, 1]:
            pr = float(pr)
            salida.append(Prediccion(
                tarea="clasificacion", modelo=nombre, probabilidad_hit=round(pr, 3), umbral=umbral,
                promocionar=bool(pr >= umbral),
                explicacion=f"probabilidad de hit {pr:.2f} {'supera' if pr >= umbral else 'no supera'} el umbral {umbral:g}"))
    return salida


@app.post("/predecir", response_model=Prediccion)
def predecir(cancion: Cancion):
    """Predicción para una canción, con la decisión que sostiene."""
    return predecir_lote([cancion])[0]


@app.post("/predecir_lote", response_model=List[Prediccion])
def predecir_varias(canciones: List[Cancion]):
    """Predicción para varias canciones (máximo 1.000 por petición)."""
    if len(canciones) > 1000:
        raise HTTPException(status_code=413, detail="Máximo 1.000 canciones por petición")
    return predecir_lote(canciones)
