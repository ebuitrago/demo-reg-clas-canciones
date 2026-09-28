"""Cliente en lote: envía las canciones de 2020 a 2025 al servicio y produce la lista priorizada.

La lista se ordena por popularidad esperada (parte 1) o por probabilidad de hit (parte 2),
según la tarea del modelo que tenga cargado el servicio.

Uso, desde la raíz del repositorio:

    python -m src.consumir                                     # servicio local
    python -m src.consumir https://su-servicio.onrender.com    # servicio desplegado
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.config import ANIO_CORTE, RUTA_REPORTES, TAMANO_LOTE, UMBRAL_HIT, URL_LOCAL
from src.datos import cargar_canciones
from src.variables import COLUMNAS_ENTRADA

TIEMPO_ESPERA = 90  # segundos; el plan gratuito de Render tarda cerca de un minuto en despertar


def pedir_predicciones(url: str, canciones: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for inicio in range(0, len(canciones), TAMANO_LOTE):
        lote = canciones.iloc[inicio : inicio + TAMANO_LOTE][COLUMNAS_ENTRADA].to_dict(orient="records")
        respuesta = requests.post(f"{url}/predecir_lote", json=lote, timeout=TIEMPO_ESPERA)
        respuesta.raise_for_status()
        filas.extend(respuesta.json())
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce la lista priorizada de promoción.")
    parser.add_argument("url", nargs="?", default=URL_LOCAL, help="URL base del servicio")
    parser.add_argument("--salida", type=Path, default=RUTA_REPORTES / "lista_promocion.csv")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    respuesta = requests.get(f"{url}/modelo", timeout=TIEMPO_ESPERA)
    respuesta.raise_for_status()
    ficha = respuesta.json()
    print(f"Modelo: {ficha['nombre']} ({ficha['tarea']}, {ficha.get('fecha')})")

    datos = cargar_canciones()
    nuevas = datos[datos["anio"] >= ANIO_CORTE].reset_index(drop=True)  # canciones que el modelo no vio
    lista = pd.concat([nuevas, pedir_predicciones(url, nuevas)], axis=1)

    puntaje = "popularidad_esperada" if ficha["tarea"] == "regresion" else "probabilidad_hit"
    lista = lista.sort_values(puntaje, ascending=False).reset_index(drop=True)
    lista.insert(0, "prioridad", range(1, len(lista) + 1))

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    lista.to_csv(args.salida, index=False)

    promocionadas = int(lista["promocionar"].sum())
    hits = lista["popularidad"] >= UMBRAL_HIT
    hits_en_lista = int((hits & lista["promocionar"]).sum())
    print(f"{len(lista)} canciones evaluadas, {promocionadas} recomendadas para promoción")
    print(
        f"De las {int(hits.sum())} que alcanzaron {UMBRAL_HIT} puntos, {hits_en_lista} estaban en la lista "
        f"y {int(hits.sum()) - hits_en_lista} quedaron por fuera"
    )
    print(f"Lista guardada en {args.salida}")
    columnas = ["prioridad", "genero", "anio", puntaje, "promocionar", "popularidad"]
    print(lista[columnas].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
