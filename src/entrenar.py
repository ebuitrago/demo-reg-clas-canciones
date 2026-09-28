"""Entrena el modelo de regresión de la Semana 9 y escribe models/modelo.joblib con su ficha.

Reproduce la selección hecha en el notebook: regresión lineal sobre las variables disponibles
antes del lanzamiento (más duración²), entrenada con las canciones hasta 2019 y evaluada en
2020-2025. Para usar su propio modelo, cambie `construir_pipeline` y vuelva a ejecutar:

    python -m src.entrenar
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.artefacto import guardar
from src.variables import COLUMNAS_ENTRADA, derivadas

RUTA_DATOS = "data/canciones.csv"
OBJETIVO = "popularidad"
CORTE_ANIO = 2020            # entrenar con < 2020, probar con >= 2020
CORTE_PROMOCION = 65         # popularidad esperada a partir de la cual se recomienda promocionar
VARIABLES_EXCLUIDAS = {
    "reproducciones_sem1": "ocurre después del lanzamiento: no existe en el momento de decidir",
    "es_hit": "se calcula a partir del objetivo (popularidad >= 70)",
}


def construir_pipeline():
    """Cambie esta función para probar otro modelo. Debe recibir las COLUMNAS_ENTRADA crudas."""
    categoricas = ["genero"]
    numericas = [c for c in COLUMNAS_ENTRADA if c not in categoricas] + ["duracion_c2"]
    preprocesar = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ("num", StandardScaler(), numericas),
    ])
    return make_pipeline(FunctionTransformer(derivadas), preprocesar, LinearRegression())


def entrenar(ruta_datos=RUTA_DATOS, ruta_modelo="models/modelo.joblib", nombre="lineal + duración²"):
    datos = pd.read_csv(ruta_datos).sort_values("anio", kind="stable").reset_index(drop=True)
    train = datos["anio"] < CORTE_ANIO
    test = ~train
    X, y = datos[COLUMNAS_ENTRADA], datos[OBJETIVO]

    pipeline = construir_pipeline().fit(X[train], y[train])
    pred = pipeline.predict(X[test])
    residuo = y[test] - pred

    # línea base: media por género calculada solo con entrenamiento
    media_genero = y[train].groupby(datos.loc[train, "genero"]).mean()
    base = datos.loc[test, "genero"].map(media_genero)

    por_genero = (pd.DataFrame({"genero": datos.loc[test, "genero"], "abs": residuo.abs(), "res": residuo})
                  .groupby("genero").agg(mae=("abs", "mean"), sesgo=("res", "mean")).round(2))

    ficha = {
        "nombre": nombre,
        "tarea": "regresion",
        "objetivo": "popularidad (0-100) que alcanzará la canción",
        "momento_prediccion": "antes del lanzamiento: la disquera decide cuánto invertir en promoción",
        "unidad": "una canción",
        "variables_entrada": list(COLUMNAS_ENTRADA),
        "variables_excluidas": VARIABLES_EXCLUIDAS,
        "entrenamiento": {"periodo": f"{datos.loc[train, 'anio'].min()}-{datos.loc[train, 'anio'].max()}",
                          "n": int(train.sum())},
        "prueba": {"periodo": f"{datos.loc[test, 'anio'].min()}-{datos.loc[test, 'anio'].max()}",
                   "n": int(test.sum()), "tipo": "temporal"},
        "linea_base": {"descripcion": "media por género (entrenamiento)",
                       "mae": round(float(mean_absolute_error(y[test], base)), 2)},
        "desempeno": {"mae": round(float(mean_absolute_error(y[test], pred)), 2),
                      "rmse": round(float(np.sqrt(mean_squared_error(y[test], pred))), 2),
                      "r2": round(float(r2_score(y[test], pred)), 3),
                      "sesgo": round(float(residuo.mean()), 2),
                      "dentro_de_10_puntos": round(float((residuo.abs() <= 10).mean()), 3)},
        "por_genero": por_genero.to_dict(orient="index"),
        "decision": {"corte_promocion": CORTE_PROMOCION,
                     "regla": "promocionar si la popularidad esperada supera el corte",
                     "nota": "el corte lo fija quien responde por el presupuesto, no el modelo"},
        "limites": "describe asociaciones en datos sintéticos; no es evidencia causal",
    }
    return guardar(pipeline, ficha, ruta_modelo)


if __name__ == "__main__":
    entrenar()
