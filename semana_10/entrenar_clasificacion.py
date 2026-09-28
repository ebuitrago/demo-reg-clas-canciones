"""Semana 10, clasificación: ¿la canción será un hit? (es_hit = popularidad >= 70)

Mismo repositorio, mismo servicio: cambia el artefacto. La ficha lleva el umbral de decisión
y los costos de error con los que se eligió. Ejecutar desde la raíz del repositorio:

    python -m semana_10.entrenar_clasificacion
    uvicorn src.app.main:app --reload --port 8000        # el servicio detecta la tarea y devuelve probabilidades
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.artefacto import guardar
from src.variables import COLUMNAS_ENTRADA, derivadas

RUTA_DATOS = "data/canciones.csv"
CORTE_ANIO = 2020
COSTO_FN = 3          # dejar sin promoción un hit real
COSTO_FP = 1          # promocionar una canción que no será hit
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "popularidad": "es el objetivo de la semana 9 y define es_hit: no puede entrar como variable",
}


def construir_pipeline():
    categoricas = ["genero"]
    numericas = [c for c in COLUMNAS_ENTRADA if c not in categoricas] + ["duracion_c2"]
    preprocesar = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ("num", StandardScaler(), numericas),
    ])
    return make_pipeline(FunctionTransformer(derivadas), preprocesar, LogisticRegression(max_iter=1000))


def elegir_umbral(y, prob, costo_fn=COSTO_FN, costo_fp=COSTO_FP):
    """Umbral que minimiza costo = costo_fn*FN + costo_fp*FP sobre probabilidades fuera de muestra."""
    mejor = None
    for u in np.arange(0.05, 0.96, 0.01):
        pred = prob >= u
        fn = int(((y == 1) & ~pred).sum()); fp = int(((y == 0) & pred).sum())
        costo = costo_fn * fn + costo_fp * fp
        if mejor is None or costo < mejor[1]:
            mejor = (round(float(u), 2), costo)
    return mejor[0]


def entrenar(ruta_datos=RUTA_DATOS, ruta_modelo="models/modelo.joblib", nombre="logística + umbral por costos"):
    datos = pd.read_csv(ruta_datos).sort_values("anio", kind="stable").reset_index(drop=True)
    train = datos["anio"] < CORTE_ANIO
    test = ~train
    X, y = datos[COLUMNAS_ENTRADA], datos["es_hit"]

    # el umbral se elige con probabilidades fuera de muestra DENTRO de entrenamiento:
    # cada pliegue de ventana creciente entrena con el pasado y predice el bloque siguiente
    Xtr, ytr = X[train].reset_index(drop=True), y[train].reset_index(drop=True)
    prob_cv, y_cv = [], []
    for ajuste, val in TimeSeriesSplit(n_splits=4).split(Xtr):
        m = construir_pipeline().fit(Xtr.iloc[ajuste], ytr.iloc[ajuste])
        prob_cv.append(m.predict_proba(Xtr.iloc[val])[:, 1]); y_cv.append(ytr.iloc[val].to_numpy())
    umbral = elegir_umbral(np.concatenate(y_cv), np.concatenate(prob_cv))

    pipeline = construir_pipeline().fit(X[train], y[train])
    prob = pipeline.predict_proba(X[test])[:, 1]
    pred = prob >= umbral
    tn, fp, fn, tp = confusion_matrix(y[test], pred).ravel()
    base_tasa = float(y[train].mean())

    ficha = {
        "nombre": nombre,
        "tarea": "clasificacion",
        "objetivo": "es_hit: 1 si la canción alcanza 70 puntos de popularidad",
        "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
        "unidad": "una canción",
        "variables_entrada": list(COLUMNAS_ENTRADA),
        "variables_excluidas": VARIABLES_EXCLUIDAS,
        "entrenamiento": {"periodo": f"{datos.loc[train, 'anio'].min()}-{datos.loc[train, 'anio'].max()}", "n": int(train.sum())},
        "prueba": {"periodo": f"{datos.loc[test, 'anio'].min()}-{datos.loc[test, 'anio'].max()}", "n": int(test.sum()), "tipo": "temporal"},
        "linea_base": {"descripcion": "predecir siempre 'no hit'",
                       "tasa_de_hits_entrenamiento": round(base_tasa, 3),
                       "costo": int(COSTO_FN * y[test].sum())},
        "desempeno": {"precision": round(float(precision_score(y[test], pred)), 3),
                      "recall": round(float(recall_score(y[test], pred)), 3),
                      "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                      "costo": int(COSTO_FN * fn + COSTO_FP * fp),
                      "hits_reales_en_prueba": int(y[test].sum())},
        "decision": {"umbral": umbral,
                     "costos": {"falso_negativo": COSTO_FN, "falso_positivo": COSTO_FP},
                     "regla": "promocionar si la probabilidad de hit supera el umbral",
                     "como_se_eligio": "mínimo costo sobre probabilidades fuera de muestra dentro de entrenamiento",
                     "nota": "los costos los fija quien responde por el presupuesto, no el modelo"},
        "limites": "probabilidades sin calibrar; describe asociaciones en datos sintéticos",
    }
    return guardar(pipeline, ficha, ruta_modelo)


if __name__ == "__main__":
    entrenar()
