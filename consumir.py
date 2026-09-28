"""Cliente: envía las canciones de 2020-2025 al servicio y produce la lista priorizada para la disquera.

    python consumir.py                                   # servicio local
    python consumir.py https://mi-servicio.onrender.com  # servicio desplegado
"""
import os
import sys

import pandas as pd
import requests

from src.variables import COLUMNAS_ENTRADA

url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
datos = pd.read_csv("data/canciones.csv")
nuevas = datos[datos["anio"] >= 2020].reset_index(drop=True)      # las canciones que el modelo no vio

ficha = requests.get(f"{url}/modelo", timeout=90).json()
print(f"Modelo: {ficha['nombre']} ({ficha['tarea']}, {ficha.get('fecha')})")

predicciones = []
for inicio in range(0, len(nuevas), 500):
    lote = nuevas.loc[inicio:inicio + 499, COLUMNAS_ENTRADA].to_dict(orient="records")
    predicciones += requests.post(f"{url}/predecir_lote", json=lote, timeout=60).json()

lista = pd.concat([nuevas, pd.DataFrame(predicciones)], axis=1)
puntaje = "popularidad_esperada" if ficha["tarea"] == "regresion" else "probabilidad_hit"
lista = lista.sort_values(puntaje, ascending=False).reset_index(drop=True)
lista.insert(0, "prioridad", range(1, len(lista) + 1))

os.makedirs("salidas", exist_ok=True)
lista.to_csv("salidas/lista_promocion.csv", index=False)
n_promo = int(lista["promocionar"].sum())
alcanzaron = int((lista["popularidad"] >= 70).sum())
acierto = int(((lista["popularidad"] >= 70) & lista["promocionar"]).sum())
print(f"{len(lista)} canciones evaluadas, {n_promo} recomendadas para promoción")
print(f"de las {alcanzaron} que alcanzaron 70 puntos, {acierto} estaban en la lista "
      f"y {alcanzaron - acierto} se quedaron por fuera")
print("Lista guardada en salidas/lista_promocion.csv")
print(lista[["prioridad", "genero", "anio", puntaje, "promocionar", "popularidad"]].head(10).to_string(index=False))
