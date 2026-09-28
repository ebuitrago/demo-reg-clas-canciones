"""Prueba de humo contra un servicio desplegado. Uso:  python probar_servicio.py https://mi-servicio.onrender.com"""
import sys
import time

import requests

from src.artefacto import FILA_EJEMPLO

url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
fallas = 0


def paso(nombre, ok, detalle=""):
    global fallas
    fallas += 0 if ok else 1
    print(f"[{'OK' if ok else 'FALLA'}] {nombre} {detalle}")


t0 = time.time()
try:
    r = requests.get(f"{url}/salud", timeout=90)       # el plan gratuito de Render tarda ~1 min en despertar
except requests.RequestException as e:
    paso("conexión", False, f"({e})")
    sys.exit(1)
paso("salud", r.status_code == 200 and r.json().get("modelo_cargado"), f"({time.time() - t0:.0f} s)")

f = requests.get(f"{url}/modelo", timeout=30).json()
minimos = ["nombre", "tarea", "momento_prediccion", "variables_excluidas", "linea_base", "desempeno", "decision"]
paso("ficha completa", all(k in f for k in minimos), f"modelo: {f.get('nombre')} ({f.get('fecha')})")
if f.get("tarea") == "regresion":
    paso("ficha regresión", "mae" in f.get("desempeno", {}) and "corte_promocion" in f.get("decision", {}),
         f"MAE {f['desempeno'].get('mae')} vs línea base {f['linea_base'].get('mae')}")
else:
    paso("ficha clasificación", "umbral" in f.get("decision", {}), f"umbral {f['decision'].get('umbral')}")

r = requests.post(f"{url}/predecir", json=FILA_EJEMPLO, timeout=30)
paso("predicción", r.status_code == 200 and "promocionar" in r.json(), r.text[:160])
r = requests.post(f"{url}/predecir", json=dict(FILA_EJEMPLO, genero="salsa"), timeout=30)
paso("valida la entrada", r.status_code == 422)
print("\nRESULTADO:", "servicio en orden" if fallas == 0 else f"{fallas} falla(s)")
sys.exit(fallas)
