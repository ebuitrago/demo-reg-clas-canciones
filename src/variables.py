"""Variables de entrada, variables derivadas y preprocesamiento comunes a las dos partes del taller.

El servicio importa este módulo al cargar el artefacto: si un pipeline usa una función que no
está aquí, el archivo .joblib no se puede cargar. Por eso toda variable derivada se define en
este módulo y no en el notebook.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columnas que el servicio recibe en cada petición: las que existen antes del lanzamiento.
COLUMNAS_ENTRADA = [
    "bailabilidad",
    "energia",
    "valencia",
    "acustica",
    "tempo",
    "duracion_min",
    "volumen_db",
    "anio",
    "colaboracion",
    "genero",
]
CATEGORICAS = ["genero"]
GENEROS = ["pop", "urbano", "rock", "electronica", "indie"]

# Columnas del conjunto de datos que no pueden entrar al modelo en ninguna de las dos partes
# porque no existen en el momento de decidir o porque se derivan del objetivo.
COLUMNAS_POSTERIORES = ["reproducciones_sem1", "popularidad", "es_hit"]

DURACION_REFERENCIA = 3.5  # minutos; centro de la variable cuadrática

# Canción de ejemplo usada por las pruebas, la prueba de humo y la documentación.
FILA_EJEMPLO = {
    "bailabilidad": 0.72,
    "energia": 0.65,
    "valencia": 0.55,
    "acustica": 0.12,
    "tempo": 118.0,
    "duracion_min": 3.4,
    "volumen_db": -6.5,
    "anio": 2026,
    "colaboracion": 1,
    "genero": "urbano",
}


def derivadas(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas.

    Se calculan fila a fila con constantes fijadas de antemano, así que no aprenden nada de
    otros registros y pueden ir dentro del pipeline sin causar fuga.
    """
    X = X.copy()
    X["duracion_c2"] = (X["duracion_min"] - DURACION_REFERENCIA) ** 2
    return X


def preprocesamiento() -> ColumnTransformer:
    """Codificación de género y escalado de las numéricas, aplicado después de `derivadas`."""
    numericas = [c for c in COLUMNAS_ENTRADA if c not in CATEGORICAS] + ["duracion_c2"]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("num", StandardScaler(), numericas),
        ]
    )
