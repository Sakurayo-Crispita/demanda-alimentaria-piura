"""Estimación de población reportada a partir de daños en viviendas."""
import json
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from datos import (ARCHIVO_DATOS, ARCHIVO_EPISODIOS, VIVIENDAS, ENTRADAS,
                   OBJETIVO, RAIZ, huella)

RESULTADOS = RAIZ / "resultados"
ETIQUETAS = {"viviendas_afectadas": "Viviendas afectadas", "viviendas_destruidas": "Viviendas destruidas"}
CATEGORICAS = ["provincia", "peligro"]
NUMERICAS = ["mes_seno", "mes_coseno", *VIVIENDAS]
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]


def construir_variables(registros):
    datos = registros[ENTRADAS].copy()
    angulo = 2 * np.pi * (datos.pop("mes").astype(float) - 1) / 12
    # La codificación cíclica mantiene la proximidad entre diciembre y enero.
    datos["mes_seno"], datos["mes_coseno"] = np.sin(angulo), np.cos(angulo)
    return datos[CATEGORICAS + NUMERICAS]


def predecir(pipeline, registros):
    predicciones = np.maximum(pipeline.predict(construir_variables(registros)), 0)
    if not np.isfinite(predicciones).all():
        raise ValueError("El modelo produjo un resultado numérico inválido.")
    return predicciones


def cargar_modelo(carpeta=RESULTADOS):
    carpeta = Path(carpeta)
    meta = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    if meta.get("objetivo") != OBJETIVO:
        raise ValueError("El modelo no corresponde a la variable de población reportada.")
    for nombre, ruta in (("dataset", ARCHIVO_DATOS), ("episodios", ARCHIVO_EPISODIOS)):
        if meta["huellas"][nombre] != huella(ruta):
            raise ValueError("Los datos cambiaron después del entrenamiento. Ejecute entrenar.py.")
    if meta["huellas"]["modelo"] != huella(carpeta / "modelo.joblib"):
        raise ValueError("El archivo del modelo no corresponde a sus resultados.")
    # El archivo se produce localmente; no se admiten modelos cargados desde la interfaz.
    return joblib.load(carpeta / "modelo.joblib"), meta


def estimar(pipeline, meta, entrada):
    datos = {}
    for campo in CATEGORICAS:
        if entrada.get(campo) not in meta["categorias"][campo]:
            raise ValueError(f"Categoría no admitida en {campo}.")
        datos[campo] = entrada[campo]
    avisos = []
    for campo in ["mes", *VIVIENDAS]:
        valor = entrada.get(campo)
        if not isinstance(valor, Integral) or isinstance(valor, (bool, np.bool_)):
            raise ValueError(f"{campo} debe ser un entero.")
        if not 0 <= valor <= 1_000_000_000 or (campo == "mes" and not 1 <= valor <= 12):
            raise ValueError(f"Valor fuera del intervalo permitido en {campo}.")
        datos[campo] = int(valor)
        if campo != "mes":
            minimo, maximo = meta["rangos"][campo]
            if not minimo <= valor <= maximo:
                avisos.append(f"{ETIQUETAS[campo]}: fuera del rango de ajuste ({minimo}–{maximo}).")
    if [datos["provincia"], datos["peligro"]] not in meta["combinaciones"]:
        avisos.append("La combinación de provincia y peligro no apareció en los datos de ajuste.")
    return float(predecir(pipeline, pd.DataFrame([datos]))[0]), avisos
