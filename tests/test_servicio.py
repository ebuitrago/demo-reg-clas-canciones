"""Pruebas del servicio. Ejecutar con: pytest -q   (requiere models/modelo.joblib)"""
import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.artefacto import FILA_EJEMPLO


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:      # el "with" dispara el evento de arranque que carga el modelo
        yield c


def test_raiz_sirve_cliente_web(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Predicción de popularidad" in r.text


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["modelo_cargado"] is True


def test_ficha_tiene_lo_minimo(cliente):
    f = cliente.get("/modelo").json()
    for campo in ("nombre", "tarea", "momento_prediccion", "variables_excluidas", "linea_base", "desempeno", "decision"):
        assert campo in f
    assert f["tarea"] in ("regresion", "clasificacion")


def test_prediccion(cliente):
    r = cliente.post("/predecir", json=FILA_EJEMPLO)
    assert r.status_code == 200, r.text
    p = r.json()
    assert isinstance(p["promocionar"], bool)
    if p["tarea"] == "regresion":
        assert 0 <= p["popularidad_esperada"] <= 120
        assert p["rango"][0] < p["popularidad_esperada"] < p["rango"][1]
    else:
        assert 0 <= p["probabilidad_hit"] <= 1


def test_entrada_invalida(cliente):
    malo = dict(FILA_EJEMPLO, genero="salsa")
    assert cliente.post("/predecir", json=malo).status_code == 422
    malo = dict(FILA_EJEMPLO, bailabilidad=3)
    assert cliente.post("/predecir", json=malo).status_code == 422
    con_fuga = dict(FILA_EJEMPLO, reproducciones_sem1=120000)     # no existe al decidir
    assert cliente.post("/predecir", json=con_fuga).status_code == 422


def test_lote(cliente):
    r = cliente.post("/predecir_lote", json=[FILA_EJEMPLO, dict(FILA_EJEMPLO, genero="indie")])
    assert r.status_code == 200 and len(r.json()) == 2


def test_lote_demasiado_grande(cliente):
    r = cliente.post("/predecir_lote", json=[FILA_EJEMPLO] * 1001)
    assert r.status_code == 413
