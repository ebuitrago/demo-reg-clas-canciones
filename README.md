<div align="center">

# Predicción de popularidad de canciones

Repositorio base del taller en dos sprints: regresión lineal y clasificación binaria

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

</div>

---

Una disquera debe decidir, antes del lanzamiento, en qué canciones invertir promoción. Este repositorio contiene un servicio de predicción listo para publicar y el código que entrena los modelos de referencia de los dos sprints del taller.

El servicio lee un único archivo, `models/modelo.pkl`, que contiene el modelo entrenado y su ficha. **En cada sprint, su trabajo es reemplazar ese archivo por el de su propio modelo y publicar el servicio en Render.** No necesita modificar el servicio: la tarea, la regla de decisión y el error esperado se leen de la ficha.

## Los dos sprints

Cada sprint termina con un incremento funcionando: el servicio publicado con un modelo propio y una ficha que respalda sus decisiones.

| | Sprint 1: regresión lineal | Sprint 2: clasificación binaria |
|---|---|---|
| Objetivo del sprint | Publicar un servicio que estime la popularidad de una canción nueva y recomiende si promocionarla | Publicar el mismo servicio con un clasificador que estime la probabilidad de hit y decida según el costo de los errores |
| Pregunta | ¿Qué popularidad alcanzará la canción? | ¿La canción será un hit? |
| Objetivo del modelo | `popularidad` (0 a 100) | `es_hit` (1 si la popularidad llega a 70) |
| Variables excluidas | `reproducciones_sem1`, `es_hit` | `reproducciones_sem1`, `popularidad` |
| Modelo de referencia | Regresión lineal con duración² | Regresión logística |
| Resultado de referencia (2020-2025) | MAE 5,75 frente a 10,44 de la línea base | Costo 99 frente a 318 de la línea base |
| Regla de decisión | Promocionar si la popularidad esperada supera el corte (65) | Promocionar si la probabilidad de hit supera el umbral (0,27) |
| Código de entrenamiento | `src/entrenar_regresion.py` | `src/entrenar_clasificacion.py` |
| En el curso | Semana 9 | Semana 10 |

En los dos sprints se entrena con las canciones de 1995 a 2019 y se evalúa con las de 2020 a 2025. El repositorio trae el modelo de referencia del sprint 1 en `models/modelo.pkl`.

## Cómo trabajar cada sprint

1. Clone el repositorio e instale las dependencias (ver [Ejecución local](#ejecución-local)).
2. Entrene su modelo y guárdelo en `models/modelo.pkl` con su ficha (ver [Crear su modelo](#crear-su-modelo)).
3. Compruebe con `pytest -q` que el servicio lo carga y responde.
4. Suba el cambio a su repositorio en GitHub y publique el servicio en Render (ver [Publicar en Render](#publicar-en-render)).
5. Revise la definición de terminado y entregue la URL del servicio.

En el sprint 2 se repiten los pasos 2 a 5 con el clasificador, sobre el mismo repositorio, el mismo servicio y la misma URL.

### Definición de terminado

Un sprint está terminado cuando se cumplen las cuatro condiciones:

- [ ] `pytest -q` pasa con su `models/modelo.pkl`.
- [ ] La ficha (`GET /modelo`) reporta las métricas de su propia evaluación sobre 2020-2025, y su modelo supera la línea base.
- [ ] El servicio está publicado en Render con su modelo.
- [ ] `python -m src.probar_servicio SU_URL` termina con "servicio en orden". Esta es la prueba con la que se califica.

## Crear su modelo

**Opción A: desde el código del repositorio.** Edite `construir_pipeline()` y `NOMBRE` en el script del sprint y ejecútelo. El script entrena, evalúa, arma la ficha y escribe `models/modelo.pkl`.

```bash
python -m src.entrenar_regresion       # sprint 1
python -m src.entrenar_clasificacion   # sprint 2
```

**Opción B: desde su notebook.** Use las funciones del repositorio para que el servicio pueda cargar el archivo:

```python
import sys
sys.path.append("ruta/a/demo-reg-clas-canciones")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from src.variables import COLUMNAS_ENTRADA, derivadas, preprocesamiento
from src.artefacto import guardar

mi_pipeline = make_pipeline(FunctionTransformer(derivadas), preprocesamiento(), MiEstimador())
mi_pipeline.fit(X_entrenamiento[COLUMNAS_ENTRADA], y_entrenamiento)

guardar(mi_pipeline, mi_ficha, "ruta/a/demo-reg-clas-canciones/models/modelo.pkl")
```

`guardar()` revisa la ficha y prueba el pipeline antes de escribir el archivo; si falta algo, el mensaje dice qué.

### Qué debe cumplir el archivo

- **Entrada.** El pipeline recibe las 10 columnas crudas de `COLUMNAS_ENTRADA` (`src/variables.py`). El preprocesamiento va dentro del pipeline. Si crea una variable derivada nueva, agréguela en `derivadas()` de `src/variables.py` y no en el notebook, porque el servicio debe encontrarla al cargar el archivo.
- **Ficha.** Debe tener `nombre`, `tarea` (`"regresion"` o `"clasificacion"`), `objetivo`, `momento_prediccion`, `variables_excluidas`, `linea_base`, `desempeno` y `decision`.
  - En regresión, `desempeno` lleva `mae` y `decision` lleva `corte_promocion`.
  - En clasificación, `decision` lleva `umbral` y el modelo debe tener `predict_proba`.
  - Los scripts de `src/` muestran una ficha completa para cada sprint.
- **Versiones.** Entrene con las versiones de `requirements.txt`. Un `.pkl` creado con otra versión de scikit-learn puede no cargar en Render. En Colab, ejecute primero `pip install -r requirements.txt`.
- **Tamaño.** Menos de 20 MB. Un Random Forest sin límite de profundidad puede superarlo.

## API

| Método | Ruta | Respuesta |
|--------|------|-----------|
| `GET` | `/docs` | Documentación interactiva: permite probar todas las rutas desde el navegador |
| `GET` | `/salud` | Estado del servicio y modelo cargado |
| `GET` | `/modelo` | Ficha del modelo |
| `POST` | `/predecir` | Predicción y decisión para una canción |
| `POST` | `/predecir_lote` | Lo mismo para hasta 1.000 canciones |

Ejemplo de entrada para `/predecir`:

```json
{"bailabilidad": 0.72, "energia": 0.65, "valencia": 0.55, "acustica": 0.12, "tempo": 118,
 "duracion_min": 3.4, "volumen_db": -6.5, "anio": 2026, "colaboracion": 1, "genero": "urbano"}
```

Respuesta con el modelo del sprint 1:

```json
{"tarea": "regresion", "popularidad_esperada": 84.8, "rango": [79.1, 90.6], "corte_promocion": 65.0,
 "promocionar": true, "explicacion": "popularidad esperada 84.8 (±5.75) supera el corte 65"}
```

Con el modelo del sprint 2, la misma entrada devuelve `probabilidad_hit` y `umbral` en lugar de `popularidad_esperada`, `rango` y `corte_promocion`. Si la entrada trae un campo que no existe antes del lanzamiento, como `reproducciones_sem1`, el servicio responde `422`.

## Estructura

```
demo-reg-clas-canciones/
├── data/canciones.csv              # 2.000 canciones sintéticas, 1995-2025
├── models/modelo.pkl               # El archivo que usted reemplaza en cada sprint
├── src/
│   ├── variables.py                # Columnas de entrada, variables derivadas y partición temporal
│   ├── artefacto.py                # guardar() y cargar() del modelo
│   ├── entrenar_regresion.py       # Sprint 1: entrena el modelo de regresión
│   ├── entrenar_clasificacion.py   # Sprint 2: entrena el clasificador
│   ├── probar_servicio.py          # Prueba de humo contra el servicio publicado
│   └── app/
│       ├── esquemas.py             # Formato de entrada y salida
│       └── main.py                 # Servicio FastAPI
├── tests/test_servicio.py          # Pruebas
├── requirements.txt                # Dependencias con versiones fijas
├── render.yaml                     # Configuración para Render
└── .python-version                 # Python 3.11
```

## Ejecución local

Desde la raíz del repositorio:

```bash
git clone https://github.com/ebuitrago/demo-reg-clas-canciones.git
cd demo-reg-clas-canciones
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # comprueba que models/modelo.pkl carga y responde
uvicorn src.app.main:app --reload    # http://127.0.0.1:8000/docs
```

Con el servicio arriba, en otra terminal:

```bash
python -m src.probar_servicio
```

## Publicar en Render

Render tiene un plan gratuito que no pide tarjeta de crédito.

1. Cree en su cuenta de GitHub un repositorio propio con este contenido y su `models/modelo.pkl` (puede usar **Fork** o **Use this template**, o subir los archivos).
2. Entre a <https://render.com> con su cuenta de GitHub.
3. En el panel elija **New +**, luego **Blueprint**, seleccione su repositorio y confirme con **Apply**. Render lee `render.yaml` y construye el servicio en 3 a 5 minutos.
4. Copie la URL que asigna Render y compruébela:

   ```bash
   python -m src.probar_servicio https://su-servicio.onrender.com
   ```

Cada vez que suba un cambio a GitHub, Render vuelve a publicar el servicio con la misma URL. Para el sprint 2 basta con subir el nuevo `models/modelo.pkl`.

En el plan gratuito, el servicio se suspende tras 15 minutos sin uso y tarda cerca de un minuto en responder la primera petición. La prueba de humo espera hasta 90 segundos.

## Ruta del taller

| Sprint | Tema | Estado | Contenido en el repositorio |
|--------|------|--------|-----------------------------|
| 1 | Regresión e IA para analítica | Disponible | `src/entrenar_regresion.py` y el modelo de referencia en `models/modelo.pkl` |
| 2 | Clasificación binaria | Siguiente | `src/entrenar_clasificacion.py`: el mismo servicio con otro `models/modelo.pkl` |

## Problemas frecuentes

| Síntoma | Qué hacer |
|---|---|
| Aviso de que el modelo se creó con otra versión de scikit-learn | Reentrene con las versiones de `requirements.txt` |
| `503` en `/modelo` o `/predecir` | No existe `models/modelo.pkl` en el repositorio: súbalo |
| `422` en `/predecir` | La entrada no cumple el formato de `src/app/esquemas.py`; el detalle de la respuesta indica el campo |
| `AttributeError` al cargar el modelo | El pipeline usa una función que no está en `src/variables.py`: muévala allí y reentrene |
| `ModuleNotFoundError: No module named 'src'` | Ejecute los comandos desde la raíz del repositorio |
| La primera petición a Render tarda o falla | El servicio estaba suspendido; repita en un minuto |

---

<div align="center">
<sub>Universidad Central, Ingeniería de Sistemas, Data Analytics, 2026-2. Prof. Elias Buitrago Bolivar</sub>
</div>
