"""Pruebas del contrato del artefacto: el modelo del repositorio y las validaciones de guardar()."""

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.artefacto import CAMPOS_OBLIGATORIOS, cargar, guardar, validar_ficha
from src.config import RUTA_MODELO
from src.variables import COLUMNAS_ENTRADA, derivadas, preprocesamiento

FICHA_MINIMA = {
    "nombre": "prueba",
    "tarea": "regresion",
    "objetivo": "popularidad",
    "momento_prediccion": "antes del lanzamiento",
    "variables_excluidas": {"reproducciones_sem1": "posterior"},
    "linea_base": {"mae": 10.0},
    "desempeno": {"mae": 6.0},
    "decision": {"corte_promocion": 65},
}


def test_artefacto_del_repositorio_cumple_el_contrato():
    artefacto = cargar(RUTA_MODELO)
    assert artefacto["columnas_entrada"] == COLUMNAS_ENTRADA
    assert all(campo in artefacto["ficha"] for campo in CAMPOS_OBLIGATORIOS)


def test_ficha_de_regresion_supera_la_linea_base():
    ficha = cargar(RUTA_MODELO)["ficha"]
    if ficha["tarea"] != "regresion":
        pytest.skip("el artefacto cargado es de clasificación")
    assert ficha["desempeno"]["mae"] < ficha["linea_base"]["mae"]


def test_ficha_de_clasificacion_supera_la_linea_base():
    ficha = cargar(RUTA_MODELO)["ficha"]
    if ficha["tarea"] != "clasificacion":
        pytest.skip("el artefacto cargado es de regresión")
    assert ficha["desempeno"]["costo"] < ficha["linea_base"]["costo"]


@pytest.mark.parametrize("campo", CAMPOS_OBLIGATORIOS)
def test_ficha_sin_campo_obligatorio_se_rechaza(campo):
    ficha = {k: v for k, v in FICHA_MINIMA.items() if k != campo}
    with pytest.raises(ValueError):
        validar_ficha(ficha)


def test_regresion_sin_mae_se_rechaza():
    with pytest.raises(ValueError, match="mae"):
        validar_ficha({**FICHA_MINIMA, "desempeno": {"rmse": 7.0}})


def test_clasificacion_sin_umbral_se_rechaza():
    with pytest.raises(ValueError, match="umbral"):
        validar_ficha({**FICHA_MINIMA, "tarea": "clasificacion", "decision": {}})


def test_guardar_y_cargar(tmp_path, fila):
    X = pd.DataFrame([fila, {**fila, "genero": "rock", "energia": 0.2}])
    pipeline = make_pipeline(FunctionTransformer(derivadas), preprocesamiento(), DummyRegressor())
    pipeline.fit(X, [60.0, 70.0])
    ruta = tmp_path / "modelo.joblib"
    guardar(pipeline, FICHA_MINIMA, ruta)
    artefacto = cargar(ruta)
    assert artefacto["ficha"]["versiones"]["scikit_learn"]
    assert "fecha" in artefacto["ficha"]
