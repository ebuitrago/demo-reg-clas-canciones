"""Pruebas del servicio HTTP con el artefacto del repositorio, de cualquiera de las dos tareas."""

import pytest


def test_raiz_sirve_el_cliente_web(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_salud(cliente):
    cuerpo = cliente.get("/salud").json()
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["tarea"] in ("regresion", "clasificacion")


def test_modelo_devuelve_la_ficha(cliente):
    r = cliente.get("/modelo")
    assert r.status_code == 200
    assert r.json()["tarea"] in ("regresion", "clasificacion")


def test_prediccion(cliente, fila):
    r = cliente.post("/predecir", json=fila)
    assert r.status_code == 200, r.text
    p = r.json()
    assert isinstance(p["promocionar"], bool)
    if p["tarea"] == "regresion":
        assert p["rango"][0] < p["popularidad_esperada"] < p["rango"][1]
        assert p["promocionar"] == (p["popularidad_esperada"] >= p["corte_promocion"])
    else:
        assert 0 <= p["probabilidad_hit"] <= 1
        assert p["promocionar"] == (p["probabilidad_hit"] >= p["umbral"])


@pytest.mark.parametrize(
    "cambio",
    [
        {"genero": "salsa"},
        {"bailabilidad": 3},
        {"reproducciones_sem1": 120000},
        {"popularidad": 80},
        {"es_hit": 1},
    ],
)
def test_entrada_fuera_del_contrato(cliente, fila, cambio):
    assert cliente.post("/predecir", json={**fila, **cambio}).status_code == 422


def test_lote(cliente, fila):
    r = cliente.post("/predecir_lote", json=[fila, {**fila, "genero": "indie"}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_lote_demasiado_grande(cliente, fila):
    assert cliente.post("/predecir_lote", json=[fila] * 1001).status_code == 413
