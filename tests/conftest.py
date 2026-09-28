import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.variables import FILA_EJEMPLO


@pytest.fixture(scope="session")
def cliente():
    # El bloque with ejecuta el ciclo de vida de la aplicación, que carga el artefacto.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fila():
    return dict(FILA_EJEMPLO)
