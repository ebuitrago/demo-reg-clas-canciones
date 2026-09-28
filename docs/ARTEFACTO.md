# El contrato del artefacto

> Un modelo sin ficha es un número sin sustento. El artefacto empaqueta ambos para que viajen juntos: el pipeline que predice y la ficha que dice por qué se le puede creer.

## Qué es el artefacto

`models/modelo.joblib` es un diccionario serializado con `joblib`:

```python
{"pipeline": <Pipeline de scikit-learn entrenado>,
 "ficha": {...},                       # ver abajo
 "columnas_entrada": [...]}            # las de src/variables.py, en ese orden
```

`src/artefacto.py` tiene las dos únicas funciones que deben tocarlo:

| Función | Qué hace | Qué verifica |
|---|---|---|
| `guardar(pipeline, ficha, ruta)` | Escribe el artefacto | Campos obligatorios de la ficha, que el pipeline predice una fila de ejemplo, que un clasificador tiene `predict_proba`, tamaño < 20 MB. Agrega `fecha` y `versiones`. |
| `cargar(ruta)` | Lo lee al arrancar el servicio | Que estén las tres claves; avisa si la versión de scikit-learn no coincide |

Si `guardar()` rechaza su ficha, el mensaje dice exactamente qué falta. No escriba el `.joblib` con `joblib.dump` directamente: se saltaría las verificaciones y el servicio podría cargar un modelo que no puede responder.

## El pipeline

El pipeline recibe las **columnas crudas** de `src/variables.py` (`COLUMNAS_ENTRADA`) y devuelve la predicción. Todo el preprocesamiento va **dentro** del pipeline: variables derivadas, codificación de `genero`, escalado. Así lo que se entrena es exactamente lo que se despliega.

```python
Pipeline([
    ("derivadas", FunctionTransformer(derivadas)),        # src/variables.py: duracion_c2
    ("columnas", ColumnTransformer([("cat", OneHotEncoder(...), ["genero"]),
                                    ("num", StandardScaler(), numericas)])),
    ("modelo", LinearRegression()),                       # <- aquí va el suyo
])
```

Si necesita otra variable derivada, agréguela en `derivadas()` de `src/variables.py` y no en el notebook: el servicio importa esa función al cargar el artefacto, y si no la encuentra el `.joblib` no carga.

## La ficha

Es la parte analítica del entregable y lo que se califica. `GET /modelo` la devuelve tal cual. Campos obligatorios (los exige `guardar()`):

| Campo | Qué contiene | De dónde sale en su notebook |
|---|---|---|
| `nombre` | Nombre corto del modelo elegido | El renglón del registro comparativo que ganó |
| `tarea` | `"regresion"` o `"clasificacion"` | |
| `objetivo` | Qué predice y en qué unidad | Definición del problema |
| `momento_prediccion` | Cuándo se usa la predicción | Fase 1: fija qué variables están disponibles |
| `variables_excluidas` | Diccionario `{variable: razón}` | Control de fuga |
| `linea_base` | Descripción y MAE de la línea base | Fase 3 |
| `desempeno` | Para regresión, al menos `mae` (en puntos de popularidad); idealmente `rmse`, `r2`, `sesgo` | Fase 4, sobre la prueba temporal |
| `decision` | Para regresión, `corte_promocion`; para clasificación, `umbral` | Fase 5: la regla de negocio |

Campos opcionales que el modelo de referencia incluye y el cliente web muestra: `entrenamiento` y `prueba` (periodo, n, tipo), `por_genero`, `limites`. `fecha` y `versiones` los agrega `guardar()`.

El servicio usa tres valores de la ficha para responder: `desempeno.mae` (el rango ± de cada predicción), `decision.corte_promocion` (la decisión) y `nombre` + `fecha` (la etiqueta del modelo). Si el MAE de la ficha no es el de su evaluación, el rango que reporta el servicio es falso; si el corte no es el que usted eligió, la lista de promoción no es la suya.

## Cómo reemplazar el modelo por el suyo

**Opción A. Editar `src/entrenar.py`** (recomendada): cambie el estimador en `construir_pipeline()`, ajuste la ficha en `entrenar()` con sus cifras y ejecute:

```bash
python -m src.entrenar        # escribe models/modelo.joblib e imprime la ficha
pytest -q                     # las 7 pruebas deben seguir pasando
python consumir.py            # produce su lista de promoción
```

**Opción B. Guardarlo desde el notebook**, con el repositorio clonado y su entorno instalado:

```python
import sys; sys.path.insert(0, "ruta/a/prediccion-canciones")
from src.artefacto import guardar
guardar(mi_pipeline, mi_ficha, "ruta/a/prediccion-canciones/models/modelo.joblib")
```

Después, en cualquiera de las dos opciones:

```bash
git add models/modelo.joblib src/entrenar.py
git commit -m "modelo propio: <qué cambió y qué dio>"
git push                      # Render redespliega solo
```

## Reglas que no se negocian

1. **Mismas versiones.** El artefacto se crea con las versiones de `requirements.txt`. Un `.joblib` de otra versión de scikit-learn puede no cargar. En Colab: `pip install -r requirements.txt` antes de entrenar.
2. **Menos de 20 MB.** El plan gratuito de Render tiene 512 MB de memoria. Un Random Forest sin `max_depth` se pasa fácilmente; use menos árboles o limite la profundidad.
3. **Sin fuga.** `reproducciones_sem1` y `es_hit` no están en `COLUMNAS_ENTRADA` y el esquema HTTP no las acepta. Si su pipeline las necesita, el modelo está mal planteado, no el servicio.
4. **La ficha dice la verdad.** Las cifras de la ficha son las del registro comparativo de su notebook, sobre la prueba temporal. Es lo primero que se revisa.

## Cambiar de tarea

El mismo contrato sirve para clasificación: `tarea = "clasificacion"`, un pipeline con `predict_proba` y `decision.umbral` en la ficha. El servicio detecta la tarea al cargar y `/predecir` responde con `probabilidad_hit`, `umbral` y la decisión. `semana_10/` tiene el ejemplo completo.
