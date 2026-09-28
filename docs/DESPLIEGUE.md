# Guía de despliegue

Esta guía cubre la ejecución local, la publicación en Render, las alternativas si Render falla y lo que se entrega en cada parte del taller.

## Ejecución local

Todo se ejecuta desde la raíz del repositorio.

```bash
git clone https://github.com/SU_USUARIO/prediccion-canciones.git
cd prediccion-canciones
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.entrenar_regresion   # parte 1; en la parte 2: python -m src.entrenar_clasificacion
pytest
uvicorn src.app.main:app --reload --port 8000
```

Con el servicio arriba, <http://127.0.0.1:8000> abre el cliente web y <http://127.0.0.1:8000/docs> la documentación interactiva. En otra terminal, con el entorno activo:

```bash
python -m src.probar_servicio      # prueba de humo
python -m src.consumir             # escribe reportes/lista_promocion.csv
```

Si algo falla en su máquina, fallará igual en Render. Corríjalo aquí primero.

## Integración continua

Cada `git push` a `main` y cada pull request ejecutan `.github/workflows/ci.yml` en GitHub Actions: instala `requirements-dev.txt`, corre `ruff check .` y `pytest` contra el artefacto que esté en el repositorio. El resultado aparece en la pestaña Actions y junto a cada commit. Un artefacto que no carga, una ficha incompleta o un modelo que no supera su línea base dejan la verificación en rojo antes de llegar a Render.

## Publicación en Render

Render ofrece un plan gratuito para servicios web que no pide tarjeta de crédito.

1. Haga fork de este repositorio y suba su artefacto a su fork.
2. Cree una cuenta en <https://render.com> e inicie sesión con GitHub.
3. En el panel, elija New +, luego Blueprint, seleccione su fork y confirme con Apply. Render lee `render.yaml`:

   ```yaml
   runtime: python                  # PYTHON_VERSION 3.11.9
   buildCommand: pip install -r requirements.txt
   startCommand: uvicorn src.app.main:app --host 0.0.0.0 --port $PORT
   healthCheckPath: /salud
   ```

4. Espere a que termine la construcción, entre 3 y 5 minutos la primera vez. La URL tiene la forma `https://prediccion-canciones-xxxx.onrender.com`.
5. Compruebe desde su máquina:

   ```bash
   python -m src.probar_servicio https://prediccion-canciones-xxxx.onrender.com
   python -m src.consumir https://prediccion-canciones-xxxx.onrender.com
   ```

Render solo instala `requirements.txt`, que contiene las dependencias del servicio. Las herramientas de desarrollo (`pytest`, `ruff`, `requests`) están en `requirements-dev.txt` y no se instalan en producción.

Cada `git push` a `main` vuelve a desplegar el servicio. Pasar de la parte 1 a la parte 2 consiste en entrenar el clasificador y hacer push del nuevo artefacto; la URL se mantiene.

### Límites del plan gratuito

| Límite | Consecuencia |
|---|---|
| El servicio se suspende tras 15 minutos sin tráfico | La primera petición tarda cerca de un minuto y puede fallar por tiempo de espera; la segunda funciona. `probar_servicio` y `consumir` esperan hasta 90 segundos. |
| 512 MB de memoria | El artefacto debe pesar menos de 20 MB; `guardar()` lo verifica. |
| 750 horas al mes por cuenta | Alcanza para un servicio encendido todo el mes. Elimine los servicios que ya no use. |

## Alternativa: GitHub Codespaces

Si no logra desplegar en Render, abra su fork en GitHub con Code, luego Codespaces, luego Create codespace. En la terminal ejecute los comandos de la sección de ejecución local (sin crear el entorno virtual) y en la pestaña Ports cambie la visibilidad del puerto 8000 a Public. Entregue esa URL e indique en qué horario estará disponible, porque el codespace se apaga tras 30 minutos sin actividad.

Como último recurso, entregue capturas de `/`, `/docs` y `/modelo` del servicio local junto con la salida de `python -m src.probar_servicio`.

## Entrega

En cada parte del taller:

- La URL pública del servicio, que pasa `python -m src.probar_servicio`.
- El fork con el commit que introduce su artefacto y la verificación de GitHub Actions en verde.
- La ficha (`GET /modelo`) coherente con su análisis: sus cifras son las del registro comparativo.
- `reportes/registro_semana9.csv` en la parte 1, generado por su notebook.
- `reportes/lista_promocion.csv`, generada con `python -m src.consumir` contra su servicio, y dos líneas: cuántas canciones se promocionan y por qué ese corte (parte 1) o esos costos de error (parte 2).

## Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| Aviso de que el artefacto se creó con otra versión de scikit-learn | Se entrenó fuera del entorno del repositorio | Reentrene con `requirements.txt` instalado |
| `503 No hay modelo cargado` | No existe `models/modelo.joblib` o no se subió | Ejecute el script de entrenamiento y luego `git add models/` |
| `422 Unprocessable Entity` | La entrada no cumple el contrato: género desconocido, valor fuera de rango o campo adicional | Revise `src/app/esquemas.py`; el detalle de la respuesta indica el campo |
| `413` en `/predecir_lote` | Más de 1.000 canciones en una petición | Envíe por tandas, como hace `src/consumir.py` |
| `ModuleNotFoundError: No module named 'src'` | Se ejecutó desde otra carpeta | Ejecute desde la raíz con `python -m src...` |
| La primera petición a Render falla | El servicio estaba suspendido | Repita al cabo de un minuto |
| La construcción en Render falla | Versión de Python distinta o dependencia no declarada | Conserve `.python-version` y agregue la dependencia a `requirements.txt` con versión fija |
| `AttributeError` al cargar el `.joblib` | El pipeline usa una función que no está en `src/variables.py` | Muévala a ese módulo y reentrene |
| GitHub Actions en rojo en `test_artefacto` | La ficha no cumple el contrato o el modelo no supera la línea base | Lea el mensaje de la prueba y revise la ficha |
