"""Conversión de población estimada a requerimientos alimentarios normativos."""
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

FUENTE = "https://bvpad.indeci.gob.pe/doc/folletos/2017/Manual%20para%20la%20AAE%20Peru.pdf"
DIAS_EVALUACION = 7
RACIONES = [
    ("Cereales", "kg", .300),
    ("Menestras", "kg", .150),
    ("Azúcar", "kg", .040),
    ("Aceite vegetal", "litros", .040),
    ("Conserva de origen animal", "kg", .060),
]


def calcular_requerimientos(personas, dias=DIAS_EVALUACION):
    """Escenario de cobertura total; cuadro 7 y fórmula Q = (D + A) × d × R."""
    if (not isinstance(personas, Real) or isinstance(personas, (bool, np.bool_))
            or not math.isfinite(personas) or not 0 <= personas <= 1_000_000_000):
        raise ValueError("La población debe ser un número finito no negativo.")
    if not isinstance(dias, Integral) or isinstance(dias, (bool, np.bool_)) or not 1 <= dias <= 10:
        raise ValueError("El escenario de asistencia cruda admite entre 1 y 10 días.")
    planificadas = math.ceil(personas)
    return pd.DataFrame([
        {"recurso": nombre, "unidad": unidad, "personas_planificadas": planificadas,
         "dias": int(dias), "racion_persona_dia": racion,
         "cantidad": round(planificadas * int(dias) * racion, 3), "tipo": "requerimiento_normativo"}
        for nombre, unidad, racion in RACIONES
    ])


def exportar_metricas_demanda(salida, detalle, meta):
    """El error de población se propaga a cantidades derivadas; no es validación de solicitudes."""
    observado = np.asarray(detalle.observado, dtype=float)
    estimado = np.ceil(np.asarray(detalle.estimado, dtype=float))
    referencia = np.ceil(np.asarray(detalle.referencia, dtype=float))
    filas = []
    for recurso, unidad, racion in RACIONES:
        factor = DIAS_EVALUACION * racion
        error = (estimado - observado) * factor
        filas.append({"recurso": recurso, "unidad": unidad, "dias": DIAS_EVALUACION,
                      "n": len(observado), "MAE_normativo": float(np.abs(error).mean()),
                      "RMSE_normativo": float(np.sqrt(np.mean(error ** 2))),
                      "MAE_referencia_normativo": float(np.abs(referencia - observado).mean() * factor),
                      "referencia_comparacion": "poblacion_reportada_por_racion_oficial"})
    pd.DataFrame(filas).to_csv(salida / "metricas_requerimientos.csv", index=False, encoding="utf-8-sig")
    meta["demanda"] = {"tipo": "normativa", "dias_evaluacion": DIAS_EVALUACION,
                        "fuente": FUENTE, "tabla": "Cuadro 7, página impresa 10",
                        "formula": "Q = población planificada × días × ración por persona y día",
                        "alcance": "Cobertura total de las personas reportadas; no observaciones de demanda alimentaria."}
    # El resumen se completa después de calcular las métricas derivadas.
    import json
    (salida / "resumen.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
