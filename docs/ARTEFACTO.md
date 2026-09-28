# Contrato del artefacto

El artefacto es el archivo que conecta el análisis con el servicio. Empaqueta el pipeline entrenado junto con su ficha, de modo que la predicción y la evidencia que la respalda viajen juntas. Este documento describe su estructura, lo que exige y cómo reemplazarlo en cada parte del taller.

## Estructura

`models/modelo.joblib` es un diccionario serializado con `joblib`:

```python
{
    "pipeline": ...,           # Pipeline de scikit-learn entrenado
    "ficha": {...},            # descripción y evidencia del modelo
    "columnas_entrada": [...], # COLUMNAS_ENTRADA de src/variables.py, en ese orden
}
```

Solo dos funciones de `src/artefacto.py` deben leerlo o escribirlo:

| Función | Qué hace | Qué verifica |
|---|---|---|
| `guardar(pipeline, ficha, ruta)` | Escribe el artefacto | Campos obligatorios de la ficha según la tarea; que el pipeline prediga una fila de ejemplo; que un clasificador tenga `predict_proba`; tamaño menor de 20 MB. Agrega `fecha` y `versiones`. |
| `cargar(ruta)` | Lo lee al arrancar el servicio | Las tres claves, la ficha y la versión de scikit-learn; avisa si la versión no coincide |

Si `guardar()` rechaza una ficha, el mensaje indica qué falta. Escribir el archivo con `joblib.dump` directamente omite estas verificaciones y puede dejar en producción un modelo que no responde.

## Pipeline

El pipeline recibe las columnas crudas de `COLUMNAS_ENTRADA` y devuelve la predicción. Todo el preprocesamiento va dentro: variables derivadas, codificación de `genero` y escalado. Así lo que se evalúa en el notebook es exactamente lo que se despliega.

```python
make_pipeline(
    FunctionTransformer(derivadas),  # src/variables.py: agrega duracion_c2
    preprocesamiento(),              # src/variables.py: one-hot de genero y escalado
    LinearRegression(),              # parte 1; en la parte 2, LogisticRegression(max_iter=1000)
)
```

Una variable derivada nueva se agrega en `derivadas()` de `src/variables.py`. Si se define en el notebook, el servicio no la encuentra al cargar el `.joblib` y el artefacto no carga.

## Ficha

La ficha es lo que `GET /modelo` devuelve y lo primero que se revisa en la entrega. Campos obligatorios en las dos tareas:

| Campo | Contenido | Origen en el notebook |
|---|---|---|
| `nombre` | Nombre corto del modelo | La corrida seleccionada en el registro comparativo |
| `tarea` | `"regresion"` o `"clasificacion"` | |
| `objetivo` | Qué predice y en qué unidad | Planteamiento del problema |
| `momento_prediccion` | Cuándo se usa la predicción | Define qué variables están disponibles |
| `variables_excluidas` | Diccionario `{variable: razón}` | Control de fuga |
| `linea_base` | Descripción y métrica de la línea base | Registro comparativo |
| `desempeno` | Métricas sobre la prueba temporal | Registro comparativo |
| `decision` | Regla de negocio | Conclusión |

Campos adicionales que exige cada tarea:

| Tarea | En `desempeno` | En `decision` | En el pipeline |
|---|---|---|---|
| Regresión | `mae`, en puntos de popularidad | `corte_promocion` | `predict` |
| Clasificación | (recomendado: `precision`, `recall`, `matriz_confusion`, `costo`) | `umbral` (recomendado: `costos`) | `predict_proba` |

Campos opcionales que incluyen los modelos de referencia y que el cliente web muestra: `entrenamiento`, `prueba`, `por_genero` y `limites`. `fecha` y `versiones` los agrega `guardar()`.

El servicio usa tres valores de la ficha para construir cada respuesta. En regresión, `desempeno.mae` define el rango de la predicción y `decision.corte_promocion` la decisión; en clasificación, `decision.umbral` define la decisión. En ambas, `nombre` y `fecha` identifican el modelo. Si esas cifras no son las de su evaluación, el servicio reporta un rango o una decisión que su análisis no respalda.

## Reemplazar el modelo

Opción A, recomendada: edite `construir_pipeline()` y `NOMBRE` en el script de la parte correspondiente y ejecútelo.

```bash
python -m src.entrenar_regresion        # parte 1
python -m src.entrenar_clasificacion    # parte 2
pytest                                  # las pruebas deben seguir pasando
```

Los dos scripts aceptan parámetros para explorar la regla de decisión sin editar código:

```bash
python -m src.entrenar_regresion --corte 70
python -m src.entrenar_clasificacion --costo-fn 5 --costo-fp 1
```

Opción B: guarde desde el notebook, con el entorno del repositorio. La última sección de `notebooks/01_regresion_popularidad.ipynb` lo hace y compara el MAE de la ficha con el del registro.

```python
from src.artefacto import guardar
guardar(mi_pipeline, mi_ficha)          # escribe models/modelo.joblib
```

En cualquiera de las dos opciones, confirme el cambio en git con un mensaje que diga qué modelo es y qué resultado obtuvo:

```bash
git add models/modelo.joblib src/
git commit -m "Parte 1: lineal + duración², MAE 5.75 frente a 10.44 de la línea base"
git push
```

## Requisitos

1. Mismas versiones. El artefacto se crea con las versiones de `requirements.txt`; un `.joblib` de otra versión de scikit-learn puede no cargar. En Colab, instale primero `pip install -r requirements.txt`.
2. Menos de 20 MB. El plan gratuito de Render tiene 512 MB de memoria. Un Random Forest sin `max_depth` supera el límite con facilidad; use menos árboles o limite la profundidad.
3. Sin fuga. `reproducciones_sem1`, `popularidad` y `es_hit` no están en `COLUMNAS_ENTRADA` y el esquema HTTP las rechaza. Si un modelo las necesita, el problema está en el planteamiento.
4. Ficha verificable. Las cifras de la ficha son las del registro comparativo sobre la prueba temporal. `tests/test_artefacto.py` comprueba además que el modelo supere su línea base.
