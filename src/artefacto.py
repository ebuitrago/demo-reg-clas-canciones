"""Guardar y cargar el artefacto del modelo: un Pipeline de scikit-learn junto con su ficha.

El artefacto es un diccionario serializado con joblib:
    {"pipeline": <Pipeline entrenado>, "ficha": {...}, "columnas_entrada": [...]}

La ficha es la parte analítica del entregable: dice qué predice el modelo, con qué datos,
qué tan bien lo hace y qué decisión sostiene. Reemplazar el modelo sin actualizar la ficha
deja un modelo sin sustento; por eso `guardar` exige los campos mínimos.
"""
import json
import os
import platform
from datetime import date

import joblib
import pandas as pd
import sklearn

from src.variables import COLUMNAS_ENTRADA

TAMANO_MAXIMO_MB = 20          # el plan gratuito de Render tiene 512 MB de memoria
CAMPOS_OBLIGATORIOS = {
    "regresion": ["nombre", "tarea", "objetivo", "momento_prediccion", "variables_excluidas",
                  "linea_base", "desempeno", "decision"],
    "clasificacion": ["nombre", "tarea", "objetivo", "momento_prediccion", "variables_excluidas",
                      "linea_base", "desempeno", "decision"],
}


def guardar(pipeline, ficha: dict, ruta: str = "models/modelo.joblib") -> dict:
    """Valida la ficha, hace una predicción de prueba y escribe el artefacto."""
    tarea = ficha.get("tarea")
    if tarea not in CAMPOS_OBLIGATORIOS:
        raise ValueError("ficha['tarea'] debe ser 'regresion' o 'clasificacion'")
    faltan = [c for c in CAMPOS_OBLIGATORIOS[tarea] if c not in ficha]
    if faltan:
        raise ValueError(f"La ficha no tiene los campos obligatorios: {faltan}")
    if tarea == "regresion" and "mae" not in ficha["desempeno"]:
        raise ValueError("ficha['desempeno'] debe incluir 'mae' (en puntos de popularidad)")
    if tarea == "clasificacion" and "umbral" not in ficha["decision"]:
        raise ValueError("ficha['decision'] debe incluir 'umbral'")

    ficha = dict(ficha)
    ficha.setdefault("fecha", date.today().isoformat())
    ficha["versiones"] = {"python": platform.python_version(),
                          "scikit_learn": sklearn.__version__, "pandas": pd.__version__}

    # el pipeline debe aceptar una fila con las columnas de entrada
    ejemplo = pd.DataFrame([FILA_EJEMPLO])[COLUMNAS_ENTRADA]
    pipeline.predict(ejemplo)
    if tarea == "clasificacion" and not hasattr(pipeline, "predict_proba"):
        raise ValueError("Un modelo de clasificación debe tener predict_proba")

    artefacto = {"pipeline": pipeline, "ficha": ficha, "columnas_entrada": list(COLUMNAS_ENTRADA)}
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    joblib.dump(artefacto, ruta, compress=3)
    mb = os.path.getsize(ruta) / 1e6
    if mb > TAMANO_MAXIMO_MB:
        raise ValueError(f"El artefacto pesa {mb:.1f} MB; el límite es {TAMANO_MAXIMO_MB} MB. "
                         "Reduzca el modelo (menos árboles, menor profundidad).")
    print(f"Artefacto guardado en {ruta} ({mb:.1f} MB)")
    print(json.dumps(ficha, ensure_ascii=False, indent=2))
    return artefacto


def cargar(ruta: str = "models/modelo.joblib") -> dict:
    """Carga el artefacto y comprueba que las versiones coincidan con las del entorno."""
    artefacto = joblib.load(ruta)
    for clave in ("pipeline", "ficha", "columnas_entrada"):
        if clave not in artefacto:
            raise ValueError(f"El artefacto no tiene la clave '{clave}'. Use src.artefacto.guardar.")
    v = artefacto["ficha"].get("versiones", {})
    if v.get("scikit_learn") != sklearn.__version__:
        print(f"ADVERTENCIA: el artefacto se creó con scikit-learn {v.get('scikit_learn')} y el "
              f"entorno tiene {sklearn.__version__}. Reentrene con las versiones de requirements.txt.")
    return artefacto


FILA_EJEMPLO = {
    "bailabilidad": 0.72, "energia": 0.65, "valencia": 0.55, "acustica": 0.12, "tempo": 118.0,
    "duracion_min": 3.4, "volumen_db": -6.5, "anio": 2026, "colaboracion": 1, "genero": "urbano",
}
