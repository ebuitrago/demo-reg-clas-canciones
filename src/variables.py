"""Variables derivadas que el modelo espera.

Se calculan fila a fila y con constantes fijadas de antemano, así que no aprenden nada de
los demás registros y pueden aplicarse dentro del Pipeline, tanto al entrenar como al predecir.
Si su modelo usa otras variables derivadas, agréguelas aquí: el servicio importa este módulo
para poder cargar el artefacto.
"""
import pandas as pd

# columnas que el servicio recibe en cada petición (las que existen antes del lanzamiento)
COLUMNAS_ENTRADA = [
    "bailabilidad", "energia", "valencia", "acustica", "tempo",
    "duracion_min", "volumen_db", "anio", "colaboracion", "genero",
]
GENEROS = ["pop", "urbano", "rock", "electronica", "indie"]


def derivadas(X: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas a partir de las columnas de entrada."""
    X = X.copy()
    X["duracion_c2"] = (X["duracion_min"] - 3.5) ** 2
    return X
