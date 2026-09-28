"""Servicio de predicción: carga el artefacto al arrancar y expone la ficha y las predicciones.

    uvicorn src.app.main:app --reload --port 8000

El servicio no contiene reglas de negocio propias: el corte de promoción o el umbral, el
error esperado y la tarea (regresión o clasificación) se leen de la ficha del artefacto.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from src import __version__
from src.app.esquemas import Cancion, Prediccion, Salud
from src.artefacto import cargar
from src.config import RUTA_MODELO, RUTA_PUBLIC

logger = logging.getLogger("uvicorn.error")

MAXIMO_LOTE = 1000


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga el artefacto una sola vez, al arrancar el servicio."""
    app.state.artefacto = None
    if RUTA_MODELO.exists():
        app.state.artefacto = cargar(RUTA_MODELO)
        ficha = app.state.artefacto["ficha"]
        logger.info("Modelo cargado: %s (%s, %s)", ficha["nombre"], ficha["tarea"], ficha.get("fecha"))
    else:
        logger.warning("No existe %s. Ejecute python -m src.entrenar_regresion", RUTA_MODELO)
    yield


app = FastAPI(
    title="Predicción de popularidad de canciones",
    description="Servicio del taller de regresión lineal y clasificación binaria (Data Analytics).",
    version=__version__,
    lifespan=ciclo_de_vida,
)


def artefacto_cargado(request: Request) -> dict:
    artefacto = request.app.state.artefacto
    if artefacto is None:
        raise HTTPException(
            status_code=503,
            detail=f"No hay modelo cargado en {RUTA_MODELO}. Entrene uno y reinicie el servicio.",
        )
    return artefacto


def etiqueta(ficha: dict) -> str:
    return f"{ficha['nombre']} ({ficha.get('fecha', 'sin fecha')})"


def predecir_filas(artefacto: dict, canciones: list[Cancion]) -> list[Prediccion]:
    ficha, pipeline = artefacto["ficha"], artefacto["pipeline"]
    X = pd.DataFrame([c.model_dump() for c in canciones])[artefacto["columnas_entrada"]]
    modelo = etiqueta(ficha)

    if ficha["tarea"] == "regresion":
        mae = float(ficha["desempeno"]["mae"])
        corte = float(ficha["decision"]["corte_promocion"])
        salida = []
        for valor in pipeline.predict(X):
            valor = float(valor)
            promocionar = valor >= corte
            salida.append(
                Prediccion(
                    tarea="regresion",
                    modelo=modelo,
                    popularidad_esperada=round(valor, 1),
                    rango=[round(valor - mae, 1), round(valor + mae, 1)],
                    corte_promocion=corte,
                    promocionar=promocionar,
                    explicacion=(
                        f"popularidad esperada {valor:.1f} (±{mae:g}) "
                        f"{'supera' if promocionar else 'no supera'} el corte {corte:g}"
                    ),
                )
            )
        return salida

    umbral = float(ficha["decision"]["umbral"])
    salida = []
    for prob in pipeline.predict_proba(X)[:, 1]:
        prob = float(prob)
        promocionar = prob >= umbral
        salida.append(
            Prediccion(
                tarea="clasificacion",
                modelo=modelo,
                probabilidad_hit=round(prob, 3),
                umbral=umbral,
                promocionar=promocionar,
                explicacion=(
                    f"probabilidad de hit {prob:.2f} "
                    f"{'supera' if promocionar else 'no supera'} el umbral {umbral:g}"
                ),
            )
        )
    return salida


@app.get("/", include_in_schema=False)
def cliente_web() -> FileResponse:
    """Cliente web. La documentación interactiva está en /docs."""
    return FileResponse(RUTA_PUBLIC / "index.html")


@app.get("/salud", response_model=Salud)
def salud(request: Request) -> Salud:
    artefacto = request.app.state.artefacto
    cargado = artefacto is not None
    return Salud(
        estado="ok" if cargado else "sin_modelo",
        modelo_cargado=cargado,
        ruta_modelo=str(RUTA_MODELO.name),
        modelo=etiqueta(artefacto["ficha"]) if cargado else None,
        tarea=artefacto["ficha"]["tarea"] if cargado else None,
    )


@app.get("/modelo")
def modelo(request: Request) -> dict:
    """Ficha del modelo: qué predice, con qué datos, qué tan bien y qué decisión sostiene."""
    return artefacto_cargado(request)["ficha"]


@app.post("/predecir", response_model=Prediccion)
def predecir(cancion: Cancion, request: Request) -> Prediccion:
    """Predicción y decisión para una canción."""
    return predecir_filas(artefacto_cargado(request), [cancion])[0]


@app.post("/predecir_lote", response_model=list[Prediccion])
def predecir_lote(canciones: list[Cancion], request: Request) -> list[Prediccion]:
    """Predicción y decisión para varias canciones, hasta 1.000 por petición."""
    if len(canciones) > MAXIMO_LOTE:
        raise HTTPException(status_code=413, detail=f"Máximo {MAXIMO_LOTE} canciones por petición")
    return predecir_filas(artefacto_cargado(request), canciones)
