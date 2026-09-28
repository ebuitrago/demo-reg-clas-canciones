"""Configuración del proyecto: rutas y parámetros del análisis en un solo lugar.

Las rutas se resuelven desde la raíz del repositorio, de modo que los módulos funcionan
sin importar desde qué carpeta se ejecuten. Los parámetros de negocio (corte de promoción,
costos de error) se documentan en la ficha de cada modelo.
"""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def _ruta(valor: str | os.PathLike) -> Path:
    ruta = Path(valor)
    return ruta if ruta.is_absolute() else RAIZ / ruta


# Rutas
RUTA_DATOS = RAIZ / "data" / "canciones.csv"
RUTA_MODELO = _ruta(os.environ.get("RUTA_MODELO", "models/modelo.joblib"))
RUTA_PUBLIC = RAIZ / "public"
RUTA_REPORTES = RAIZ / "reportes"  # entregables generados: registro comparativo y lista de promoción

# Reproducibilidad
SEMILLA = 42

# Partición temporal: se entrena con anio < ANIO_CORTE y se evalúa con anio >= ANIO_CORTE
ANIO_CORTE = 2020

# Parte 1, regresión: popularidad esperada a partir de la cual se recomienda promocionar
CORTE_PROMOCION = 65

# Parte 2, clasificación: definición de hit y costo relativo de cada error
UMBRAL_HIT = 70
COSTO_FALSO_NEGATIVO = 3  # dejar sin promoción una canción que sí fue hit
COSTO_FALSO_POSITIVO = 1  # promocionar una canción que no fue hit

# Clientes
URL_LOCAL = "http://127.0.0.1:8000"
TAMANO_LOTE = 500  # canciones por petición a /predecir_lote (el servicio acepta hasta 1.000)
