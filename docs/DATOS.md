# Diccionario de datos

`data/canciones.csv` contiene 2.000 canciones sintéticas lanzadas entre 1995 y 2025, una por fila. Los datos se generaron para el curso: no describen canciones ni personas reales. No hay valores nulos ni filas duplicadas; `tests/test_datos.py` lo comprueba en cada ejecución de las pruebas.

## Columnas

| Columna | Tipo | Rango observado | Descripción | ¿Existe al decidir? | Uso |
|---|---|---|---|---|---|
| `bailabilidad` | real | 0,019 a 0,994 | Qué tan apta es la canción para bailar | Sí | Entrada |
| `energia` | real | 0,030 a 0,995 | Intensidad y actividad percibida | Sí | Entrada |
| `valencia` | real | 0,016 a 0,976 | Carácter emocional, de triste (0) a alegre (1) | Sí | Entrada |
| `acustica` | real | 0,001 a 0,948 | Grado en que la canción es acústica | Sí | Entrada |
| `tempo` | real | 60 a 200 | Pulsos por minuto | Sí | Entrada |
| `duracion_min` | real | 1,50 a 6,15 | Duración en minutos | Sí | Entrada |
| `volumen_db` | real | -18,8 a 0,0 | Sonoridad promedio en decibelios | Sí | Entrada |
| `anio` | entero | 1995 a 2025 | Año de lanzamiento | Sí | Entrada y partición temporal |
| `colaboracion` | entero | 0 o 1 | 1 si la canción es una colaboración (25,8 % de las filas) | Sí | Entrada |
| `genero` | texto | 5 categorías | `pop` (616), `urbano` (511), `rock` (333), `electronica` (289), `indie` (251) | Sí | Entrada |
| `reproducciones_sem1` | entero | 18.475 a 1.121.386 | Reproducciones durante la primera semana después del lanzamiento | No | Excluida en las dos partes |
| `popularidad` | real | 11,6 a 100 | Popularidad alcanzada, de 0 a 100 | No | Objetivo de la parte 1; excluida en la parte 2 |
| `es_hit` | entero | 0 o 1 | 1 si `popularidad >= 70` (20,6 % de las filas) | No | Objetivo de la parte 2; excluida en la parte 1 |

Las diez columnas de entrada están en `COLUMNAS_ENTRADA` (`src/variables.py`) y las tres que no existen en el momento de decidir, en `COLUMNAS_POSTERIORES`. El esquema del servicio (`src/app/esquemas.py`) admite rangos algo más amplios que los observados para aceptar canciones nuevas, por ejemplo `tempo` entre 40 y 250.

## Variable derivada

| Variable | Definición | Dónde se calcula |
|---|---|---|
| `duracion_c2` | `(duracion_min - 3,5)²` | `derivadas()` en `src/variables.py`, dentro del pipeline |

Se calcula fila a fila con una constante fijada de antemano, así que no aprende nada de otros registros y no introduce fuga. El servicio no la recibe: la calcula el pipeline.

## Partición

| Conjunto | Años | Canciones | Uso |
|---|---|---|---|
| Entrenamiento | 1995 a 2019 | 1.585 | Ajuste, validación interna con ventana creciente y elección del umbral en la parte 2 |
| Prueba | 2020 a 2025 | 415 | Evaluación final; no interviene en ninguna decisión del modelo |

El corte está en `ANIO_CORTE` (`src/config.py`). La partición es temporal porque la disquera siempre predice canciones que todavía no se han lanzado.
