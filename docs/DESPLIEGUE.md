# Guía de despliegue en Render

> El servicio que funciona en su portátil no existe para nadie más. Publicarlo es lo que convierte el modelo en algo que otros pueden usar.

## Antes de desplegar: que funcione localmente

```bash
git clone https://github.com/SU_USUARIO/prediccion-canciones.git
cd prediccion-canciones
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.entrenar                               # models/modelo.joblib con su ficha
pytest -q                                            # 7 pruebas en verde
uvicorn src.app.main:app --reload --port 8000
```

Abra <http://127.0.0.1:8000>: el cliente web muestra la ficha y permite predecir una canción. En <http://127.0.0.1:8000/docs> está la documentación interactiva. En otra terminal:

```bash
python probar_servicio.py                            # prueba de humo, la misma de la calificación
python consumir.py                                   # salidas/lista_promocion.csv
```

Si algo de esto falla en su máquina, fallará igual en Render. Arréglelo aquí primero.

## Desplegar en Render (gratis, sin tarjeta)

1. Haga **fork** de este repositorio en su cuenta de GitHub y suba allí su artefacto (`git push`).
2. Cree una cuenta en <https://render.com> iniciando sesión con GitHub.
3. En el panel: **New +** -> **Blueprint** -> elija su fork -> **Apply**. Render lee `render.yaml`:

   ```yaml
   runtime: python            # PYTHON_VERSION 3.11.9
   buildCommand: pip install -r requirements.txt
   startCommand: uvicorn src.app.main:app --host 0.0.0.0 --port $PORT
   healthCheckPath: /salud
   ```

4. Espere el build (3 a 5 minutos la primera vez). La URL queda como `https://prediccion-canciones-XXXX.onrender.com`.
5. Compruebe desde su máquina:

   ```bash
   python probar_servicio.py https://prediccion-canciones-XXXX.onrender.com
   python consumir.py https://prediccion-canciones-XXXX.onrender.com
   ```

Cada `git push` a la rama principal vuelve a desplegar el servicio. No hay que tocar nada en Render.

## Lo que hay que saber del plan gratuito

| Límite | Qué implica |
|---|---|
| Se duerme tras 15 min sin tráfico | La primera petición tarda cerca de un minuto y puede fallar por tiempo de espera; la segunda funciona. `probar_servicio.py` espera hasta 90 s. |
| 512 MB de memoria | El artefacto debe pesar menos de 20 MB (`guardar()` lo verifica). |
| 750 horas al mes por cuenta | Suficiente para un servicio. Borre los que ya no use. |
| Sin tarjeta de crédito | Basta la cuenta de GitHub. |

## Plan B: GitHub Codespaces

Si no logra desplegar en Render: abra el repositorio en GitHub con **Code -> Codespaces -> Create codespace**, ejecute en la terminal los mismos comandos de la sección local (sin `venv`) y haga público el puerto 8000 en la pestaña **Ports** -> **Port visibility** -> **Public**. Entregue esa URL. El codespace se apaga tras 30 minutos sin actividad, así que avise cuándo está disponible.

Último recurso: capturas de pantalla de `/`, `/docs` y `/modelo` del servicio local, junto con la salida de `probar_servicio.py`.

## Qué se entrega

- La **URL pública** del servicio, que pasa `probar_servicio.py`.
- El **repositorio** con su artefacto y el commit que lo introduce.
- La **ficha** (`GET /modelo`) coherente con su análisis: las cifras son las del registro comparativo de su notebook.
- La **lista priorizada** (`salidas/lista_promocion.csv`) y dos líneas: cuántas canciones se promocionan con el corte elegido y por qué ese corte.

## Errores frecuentes

| Síntoma | Causa | Qué hacer |
|---|---|---|
| `ADVERTENCIA: el artefacto se creó con scikit-learn X` | Entrenó con otra versión | Reentrene con `requirements.txt` instalado |
| `503 No hay modelo cargado` | No existe `models/modelo.joblib` o no se subió al repo | `python -m src.entrenar`, luego `git add models/` |
| `422 Unprocessable Entity` | La entrada no cumple el contrato (género fuera de la lista, valor fuera de rango) | Revise `src/app/esquemas.py`; el detalle de la respuesta dice qué campo |
| `413` en `/predecir_lote` | Más de 1.000 canciones en una petición | `consumir.py` ya envía por tandas de 500; imítelo |
| `ModuleNotFoundError: src` | Ejecutó desde otra carpeta | Todo se ejecuta desde la raíz del repositorio con `python -m ...` |
| La primera petición a Render falla | El servicio estaba dormido | Repita en un minuto |
| El build de Render falla | Versión de Python o dependencia extra | Mantenga `.python-version`; agregue lo nuevo a `requirements.txt` con versión fija |
| El `.joblib` no carga: `AttributeError ... derivadas` | El pipeline usa una función que no está en `src/variables.py` | Ponga la función allí, no en el notebook, y reentrene |
