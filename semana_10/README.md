# Semana 10, clasificación: ¿la canción será un hit?

Mismo repositorio, mismo servicio. Cambia el artefacto: ahora el objetivo es `es_hit`
(popularidad ≥ 70) y la ficha lleva el **umbral** de decisión y los **costos de error** con que se eligió.

```bash
python -m semana_10.entrenar_clasificacion     # sobrescribe models/modelo.joblib
pytest -q
uvicorn src.app.main:app --reload --port 8000  # /predecir devuelve probabilidad_hit, umbral y la decisión
python consumir.py                              # la lista se ordena ahora por probabilidad
```

Puntos para el trabajo de la semana:

- `popularidad` queda excluida como variable: define el objetivo.
- El umbral se elige con probabilidades fuera de muestra dentro de entrenamiento, minimizando
  `3*FN + FP`. Cambie los costos y observe cómo se mueve el umbral y la lista.
- Las probabilidades no están calibradas: compruébelo y decida si hace falta corregirlo antes de usar el umbral.
