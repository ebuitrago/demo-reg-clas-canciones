"""Parte 2: entrena el clasificador de es_hit, elige el umbral por costos y escribe el artefacto.

Usa las mismas variables de entrada y la misma partición temporal que la parte 1. El umbral
se elige con probabilidades fuera de muestra dentro del periodo de entrenamiento, de modo que
el periodo de prueba no interviene en ninguna decisión.

Uso, desde la raíz del repositorio:

    python -m src.entrenar_clasificacion
    python -m src.entrenar_clasificacion --costo-fn 5 --costo-fp 1

Para usar su propio modelo, cambie `construir_pipeline` y actualice `NOMBRE`.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.artefacto import guardar
from src.config import COSTO_FALSO_NEGATIVO, COSTO_FALSO_POSITIVO, RUTA_MODELO, UMBRAL_HIT
from src.datos import cargar_canciones, particion_temporal, resumen_periodo
from src.variables import COLUMNAS_ENTRADA, derivadas, preprocesamiento

NOMBRE = "logística + umbral por costos"
OBJETIVO = "es_hit"
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "popularidad": "define es_hit: usarla como entrada equivale a conocer la respuesta",
}
UMBRALES = np.round(np.arange(0.05, 0.96, 0.01), 2)


def construir_pipeline() -> Pipeline:
    """Pipeline completo. Debe recibir las COLUMNAS_ENTRADA crudas y tener predict_proba."""
    return make_pipeline(
        FunctionTransformer(derivadas), preprocesamiento(), LogisticRegression(max_iter=1000)
    )


def costo(real: np.ndarray, pred: np.ndarray, costo_fn: float, costo_fp: float) -> float:
    falsos_negativos = int(((real == 1) & ~pred).sum())
    falsos_positivos = int(((real == 0) & pred).sum())
    return costo_fn * falsos_negativos + costo_fp * falsos_positivos


def elegir_umbral(real: np.ndarray, prob: np.ndarray, costo_fn: float, costo_fp: float) -> float:
    """Umbral que minimiza el costo total de los errores; en empate, el menor."""
    costos = [costo(real, prob >= u, costo_fn, costo_fp) for u in UMBRALES]
    return float(UMBRALES[int(np.argmin(costos))])


def probabilidades_fuera_de_muestra(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Ventana creciente: cada pliegue entrena con el pasado y predice el bloque siguiente."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    reales, probs = [], []
    for ajuste, validacion in TimeSeriesSplit(n_splits=4).split(X):
        modelo = construir_pipeline().fit(X.iloc[ajuste], y.iloc[ajuste])
        probs.append(modelo.predict_proba(X.iloc[validacion])[:, 1])
        reales.append(y.iloc[validacion].to_numpy())
    return np.concatenate(reales), np.concatenate(probs)


def entrenar(
    ruta_modelo: Path = RUTA_MODELO,
    costo_fn: float = COSTO_FALSO_NEGATIVO,
    costo_fp: float = COSTO_FALSO_POSITIVO,
) -> dict:
    datos = cargar_canciones()
    entrenamiento, prueba = particion_temporal(datos)
    X, y = datos[COLUMNAS_ENTRADA], datos[OBJETIVO]

    umbral = elegir_umbral(*probabilidades_fuera_de_muestra(X[entrenamiento], y[entrenamiento]), costo_fn, costo_fp)

    pipeline = construir_pipeline().fit(X[entrenamiento], y[entrenamiento])
    pred = pipeline.predict_proba(X[prueba])[:, 1] >= umbral
    real = y[prueba].to_numpy()
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()

    ficha = {
        "nombre": NOMBRE,
        "tarea": "clasificacion",
        "objetivo": f"es_hit: 1 si la canción alcanza {UMBRAL_HIT} puntos de popularidad",
        "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
        "unidad": "una canción",
        "variables_entrada": list(COLUMNAS_ENTRADA),
        "variables_excluidas": VARIABLES_EXCLUIDAS,
        "entrenamiento": resumen_periodo(datos, entrenamiento),
        "prueba": {**resumen_periodo(datos, prueba), "tipo": "temporal"},
        "linea_base": {
            "descripcion": "predecir siempre 'no hit'",
            "tasa_de_hits_entrenamiento": round(float(y[entrenamiento].mean()), 3),
            "costo": costo(real, np.zeros_like(pred), costo_fn, costo_fp),
        },
        "desempeno": {
            "precision": round(float(precision_score(real, pred)), 3),
            "recall": round(float(recall_score(real, pred)), 3),
            "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "costo": costo(real, pred, costo_fn, costo_fp),
            "hits_reales_en_prueba": int(real.sum()),
        },
        "decision": {
            "umbral": umbral,
            "costos": {"falso_negativo": costo_fn, "falso_positivo": costo_fp},
            "regla": "promocionar si la probabilidad de hit supera el umbral",
            "como_se_eligio": "mínimo costo sobre probabilidades fuera de muestra dentro de entrenamiento",
            "nota": "los costos los fija quien responde por el presupuesto, no el modelo",
        },
        "limites": "probabilidades sin calibrar; describe asociaciones en datos sintéticos",
    }
    return guardar(pipeline, ficha, ruta_modelo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", type=Path, default=RUTA_MODELO, help="ruta del artefacto")
    parser.add_argument("--costo-fn", type=float, default=COSTO_FALSO_NEGATIVO, help="costo de un falso negativo")
    parser.add_argument("--costo-fp", type=float, default=COSTO_FALSO_POSITIVO, help="costo de un falso positivo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entrenar(args.salida, args.costo_fn, args.costo_fp)


if __name__ == "__main__":
    main()
