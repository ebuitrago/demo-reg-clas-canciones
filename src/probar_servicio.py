"""Prueba de humo contra un servicio en ejecución, local o desplegado.

Comprueba salud, ficha, una predicción y la validación de la entrada. El código de salida es
el número de fallas, de modo que puede usarse en scripts y en integración continua.

Uso, desde la raíz del repositorio:

    python -m src.probar_servicio                                   # servicio local
    python -m src.probar_servicio https://su-servicio.onrender.com  # servicio desplegado
"""

import argparse
import sys
import time

import requests

from src.artefacto import CAMPOS_OBLIGATORIOS
from src.config import URL_LOCAL
from src.variables import FILA_EJEMPLO

ESPERA_ARRANQUE = 90  # segundos; el plan gratuito de Render tarda cerca de un minuto en despertar


class Verificador:
    def __init__(self) -> None:
        self.fallas = 0

    def paso(self, nombre: str, ok: bool, detalle: str = "") -> None:
        self.fallas += 0 if ok else 1
        print(f"[{'OK' if ok else 'FALLA'}] {nombre} {detalle}".rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de humo del servicio de predicción.")
    parser.add_argument("url", nargs="?", default=URL_LOCAL, help="URL base del servicio")
    url = parser.parse_args().url.rstrip("/")
    v = Verificador()

    inicio = time.time()
    try:
        r = requests.get(f"{url}/salud", timeout=ESPERA_ARRANQUE)
    except requests.RequestException as error:
        v.paso("conexión", False, f"({error})")
        return 1
    v.paso("salud", r.status_code == 200 and r.json().get("modelo_cargado"), f"({time.time() - inicio:.0f} s)")

    ficha = requests.get(f"{url}/modelo", timeout=30).json()
    completa = all(campo in ficha for campo in CAMPOS_OBLIGATORIOS)
    v.paso("ficha completa", completa, f"modelo: {ficha.get('nombre')} ({ficha.get('fecha')})")
    if ficha.get("tarea") == "regresion":
        v.paso(
            "ficha de regresión",
            "mae" in ficha.get("desempeno", {}) and "corte_promocion" in ficha.get("decision", {}),
            f"MAE {ficha['desempeno'].get('mae')} frente a línea base {ficha['linea_base'].get('mae')}",
        )
    else:
        umbral = ficha.get("decision", {}).get("umbral")
        v.paso("ficha de clasificación", umbral is not None, f"umbral {umbral}")

    r = requests.post(f"{url}/predecir", json=FILA_EJEMPLO, timeout=30)
    v.paso("predicción", r.status_code == 200 and "promocionar" in r.json(), r.json().get("explicacion", r.text[:120]))

    r = requests.post(f"{url}/predecir", json={**FILA_EJEMPLO, "genero": "salsa"}, timeout=30)
    v.paso("rechaza un género fuera del contrato", r.status_code == 422)
    r = requests.post(f"{url}/predecir", json={**FILA_EJEMPLO, "reproducciones_sem1": 1000}, timeout=30)
    v.paso("rechaza una variable posterior al lanzamiento", r.status_code == 422)

    print("\nResultado:", "servicio en orden" if v.fallas == 0 else f"{v.fallas} falla(s)")
    return v.fallas


if __name__ == "__main__":
    sys.exit(main())
