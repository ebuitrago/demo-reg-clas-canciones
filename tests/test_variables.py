"""Pruebas de las variables de entrada y derivadas."""

import pandas as pd

from src.variables import COLUMNAS_ENTRADA, COLUMNAS_POSTERIORES, FILA_EJEMPLO, derivadas


def test_entrada_no_incluye_variables_posteriores():
    assert not set(COLUMNAS_ENTRADA) & set(COLUMNAS_POSTERIORES)


def test_derivadas_no_modifica_la_entrada():
    X = pd.DataFrame([FILA_EJEMPLO])
    resultado = derivadas(X)
    assert "duracion_c2" in resultado and "duracion_c2" not in X


def test_derivadas_es_fila_a_fila():
    # El valor de una fila no depende de las demás: no hay estadísticas aprendidas.
    X = pd.DataFrame([FILA_EJEMPLO, {**FILA_EJEMPLO, "duracion_min": 6.0}])
    assert derivadas(X)["duracion_c2"].iloc[0] == derivadas(X.iloc[[0]])["duracion_c2"].iloc[0]
