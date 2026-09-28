"""Parte 1: entrena el modelo de regresión de popularidad y escribe el artefacto con su ficha.

Reproduce el modelo seleccionado en el notebook: regresión lineal sobre las variables
disponibles antes del lanzamiento, más duración al cuadrado, entrenada con las canciones
hasta 2019 y evaluada con las de 2020 a 2025.

Uso, desde la raíz del repositorio:

    python -m src.entrenar_regresion
    python -m src.entrenar_regresion --corte 70

Para usar su propio modelo, cambie `construir_pipeline` y actualice `NOMBRE`.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.artefacto import guardar
from src.config import CORTE_PROMOCION, RUTA_MODELO
from src.datos import cargar_canciones, particion_temporal, resumen_periodo
from src.variables import COLUMNAS_ENTRADA, derivadas, preprocesamiento

NOMBRE = "lineal + duración²"
OBJETIVO = "popularidad"
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "es_hit": "se calcula a partir del objetivo (popularidad >= 70)",
}


def construir_pipeline() -> Pipeline:
    """Pipeline completo. Debe recibir las COLUMNAS_ENTRADA crudas."""
    return make_pipeline(FunctionTransformer(derivadas), preprocesamiento(), LinearRegression())


def metricas(real: pd.Series, pred: np.ndarray) -> dict:
    residuo = real - pred
    return {
        "mae": round(float(mean_absolute_error(real, pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(real, pred))), 2),
        "r2": round(float(r2_score(real, pred)), 3),
        "sesgo": round(float(residuo.mean()), 2),
        "dentro_de_10_puntos": round(float((residuo.abs() <= 10).mean()), 3),
    }


def entrenar(ruta_modelo: Path = RUTA_MODELO, corte: float = CORTE_PROMOCION) -> dict:
    datos = cargar_canciones()
    entrenamiento, prueba = particion_temporal(datos)
    X, y = datos[COLUMNAS_ENTRADA], datos[OBJETIVO]

    pipeline = construir_pipeline().fit(X[entrenamiento], y[entrenamiento])
    pred = pipeline.predict(X[prueba])

    # Línea base: media por género calculada solo con el periodo de entrenamiento.
    media_genero = y[entrenamiento].groupby(datos.loc[entrenamiento, "genero"]).mean()
    base = datos.loc[prueba, "genero"].map(media_genero)

    residuo = y[prueba] - pred
    por_genero = (
        pd.DataFrame({"genero": datos.loc[prueba, "genero"], "abs": residuo.abs(), "res": residuo})
        .groupby("genero")
        .agg(mae=("abs", "mean"), sesgo=("res", "mean"))
        .round(2)
    )

    ficha = {
        "nombre": NOMBRE,
        "tarea": "regresion",
        "objetivo": "popularidad (0-100) que alcanzará la canción",
        "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
        "unidad": "una canción",
        "variables_entrada": list(COLUMNAS_ENTRADA),
        "variables_excluidas": VARIABLES_EXCLUIDAS,
        "entrenamiento": resumen_periodo(datos, entrenamiento),
        "prueba": {**resumen_periodo(datos, prueba), "tipo": "temporal"},
        "linea_base": {
            "descripcion": "media por género (entrenamiento)",
            "mae": round(float(mean_absolute_error(y[prueba], base)), 2),
        },
        "desempeno": metricas(y[prueba], pred),
        "por_genero": por_genero.to_dict(orient="index"),
        "decision": {
            "corte_promocion": corte,
            "regla": "promocionar si la popularidad esperada supera el corte",
            "nota": "el corte lo fija quien responde por el presupuesto, no el modelo",
        },
        "limites": "describe asociaciones en datos sintéticos; no es evidencia causal",
    }
    return guardar(pipeline, ficha, ruta_modelo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, default=RUTA_MODELO, help="ruta del artefacto")
    parser.add_argument("--corte", type=float, default=CORTE_PROMOCION, help="corte de promoción")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entrenar(args.salida, args.corte)


if __name__ == "__main__":
    main()
