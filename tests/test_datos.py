"""Pruebas del conjunto de datos: estructura, rangos y coherencia del objetivo."""

import pytest

from src.app.esquemas import Cancion
from src.config import ANIO_CORTE, UMBRAL_HIT
from src.datos import cargar_canciones, particion_temporal
from src.variables import COLUMNAS_ENTRADA, GENEROS


@pytest.fixture(scope="module")
def datos():
    return cargar_canciones()


def test_sin_nulos_ni_duplicados(datos):
    assert not datos.isna().any().any()
    assert not datos.duplicated().any()


def test_generos_conocidos(datos):
    assert set(datos["genero"]) <= set(GENEROS)


def test_filas_cumplen_el_contrato_del_servicio(datos):
    # Si una fila real no pasa el esquema HTTP, el servicio no podría recibirla.
    for fila in datos[COLUMNAS_ENTRADA].to_dict(orient="records"):
        Cancion(**fila)


def test_es_hit_coincide_con_popularidad(datos):
    assert ((datos["popularidad"] >= UMBRAL_HIT) == datos["es_hit"].astype(bool)).all()


def test_particion_temporal_sin_solapamiento(datos):
    entrenamiento, prueba = particion_temporal(datos)
    assert datos.loc[entrenamiento, "anio"].max() < ANIO_CORTE <= datos.loc[prueba, "anio"].min()
    assert (entrenamiento ^ prueba).all()
