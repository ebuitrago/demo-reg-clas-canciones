"""Carga y partición de los datos, compartidas por los dos scripts de entrenamiento."""

from pathlib import Path

import pandas as pd

from src.config import ANIO_CORTE, RUTA_DATOS
from src.variables import COLUMNAS_ENTRADA, COLUMNAS_POSTERIORES


def cargar_canciones(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV, verifica que tenga las columnas esperadas y lo ordena por año."""
    datos = pd.read_csv(ruta)
    faltan = sorted(set(COLUMNAS_ENTRADA + COLUMNAS_POSTERIORES) - set(datos.columns))
    if faltan:
        raise ValueError(f"{ruta} no tiene las columnas {faltan}")
    return datos.sort_values("anio", kind="stable").reset_index(drop=True)


def particion_temporal(datos: pd.DataFrame, anio_corte: int = ANIO_CORTE) -> tuple[pd.Series, pd.Series]:
    """Máscaras de entrenamiento (anio < corte) y prueba (anio >= corte)."""
    entrenamiento = datos["anio"] < anio_corte
    return entrenamiento, ~entrenamiento


def resumen_periodo(datos: pd.DataFrame, mascara: pd.Series) -> dict:
    """Periodo y número de canciones de una partición, para la ficha."""
    anios = datos.loc[mascara, "anio"]
    return {"periodo": f"{anios.min()}-{anios.max()}", "n": int(mascara.sum())}
