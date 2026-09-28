"""Guardar y cargar el artefacto del modelo: un pipeline de scikit-learn junto con su ficha.

El artefacto es un diccionario serializado con joblib:

    {"pipeline": <Pipeline entrenado>, "ficha": {...}, "columnas_entrada": [...]}

La ficha documenta qué predice el modelo, con qué datos, qué tan bien lo hace y qué decisión
sostiene. `guardar` exige los campos mínimos y comprueba que el pipeline pueda predecir antes
de escribir el archivo; `cargar` verifica la estructura y la versión de scikit-learn.
"""

import json
import logging
import platform
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.config import RUTA_MODELO
from src.variables import COLUMNAS_ENTRADA, FILA_EJEMPLO

logger = logging.getLogger(__name__)

TAREAS = ("regresion", "clasificacion")
CAMPOS_OBLIGATORIOS = [
    "nombre",
    "tarea",
    "objetivo",
    "momento_prediccion",
    "variables_excluidas",
    "linea_base",
    "desempeno",
    "decision",
]
CLAVES_ARTEFACTO = ("pipeline", "ficha", "columnas_entrada")
TAMANO_MAXIMO_MB = 20  # el plan gratuito de Render tiene 512 MB de memoria


def validar_ficha(ficha: dict) -> None:
    """Lanza ValueError si la ficha no cumple el contrato mínimo de su tarea."""
    tarea = ficha.get("tarea")
    if tarea not in TAREAS:
        raise ValueError(f"ficha['tarea'] debe ser una de {TAREAS}, no {tarea!r}")
    faltan = [c for c in CAMPOS_OBLIGATORIOS if c not in ficha]
    if faltan:
        raise ValueError(f"La ficha no tiene los campos obligatorios: {faltan}")
    if tarea == "regresion":
        if "mae" not in ficha["desempeno"]:
            raise ValueError("ficha['desempeno'] debe incluir 'mae' (en puntos de popularidad)")
        if "corte_promocion" not in ficha["decision"]:
            raise ValueError("ficha['decision'] debe incluir 'corte_promocion'")
    if tarea == "clasificacion" and "umbral" not in ficha["decision"]:
        raise ValueError("ficha['decision'] debe incluir 'umbral'")


def guardar(pipeline, ficha: dict, ruta: Path = RUTA_MODELO) -> dict:
    """Valida la ficha, hace una predicción de prueba y escribe el artefacto."""
    validar_ficha(ficha)
    if ficha["tarea"] == "clasificacion" and not hasattr(pipeline, "predict_proba"):
        raise ValueError("Un modelo de clasificación debe tener predict_proba")

    ficha = dict(ficha)
    ficha.setdefault("fecha", date.today().isoformat())
    ficha["versiones"] = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
    }

    # El pipeline debe aceptar una fila con las columnas crudas de entrada.
    pipeline.predict(pd.DataFrame([FILA_EJEMPLO])[COLUMNAS_ENTRADA])

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    artefacto = {"pipeline": pipeline, "ficha": ficha, "columnas_entrada": list(COLUMNAS_ENTRADA)}
    joblib.dump(artefacto, ruta, compress=3)

    megas = ruta.stat().st_size / 1e6
    if megas > TAMANO_MAXIMO_MB:
        ruta.unlink()
        raise ValueError(
            f"El artefacto pesa {megas:.1f} MB y el límite es {TAMANO_MAXIMO_MB} MB. "
            "Reduzca el modelo (menos árboles, menor profundidad)."
        )
    logger.info("Artefacto guardado en %s (%.2f MB)", ruta, megas)
    logger.info(json.dumps(ficha, ensure_ascii=False, indent=2))
    return artefacto


def cargar(ruta: Path = RUTA_MODELO) -> dict:
    """Carga el artefacto y avisa si la versión de scikit-learn no coincide con la del entorno."""
    artefacto = joblib.load(ruta)
    faltan = [c for c in CLAVES_ARTEFACTO if c not in artefacto]
    if faltan:
        raise ValueError(f"El artefacto no tiene las claves {faltan}. Créelo con src.artefacto.guardar.")
    validar_ficha(artefacto["ficha"])
    version = artefacto["ficha"].get("versiones", {}).get("scikit_learn")
    if version != sklearn.__version__:
        logger.warning(
            "El artefacto se creó con scikit-learn %s y el entorno tiene %s. "
            "Reentrene con las versiones de requirements.txt.",
            version,
            sklearn.__version__,
        )
    return artefacto
