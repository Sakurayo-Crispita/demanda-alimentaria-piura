"""Comparación reproducible de tres regresores y una referencia histórica."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from demanda import exportar_metricas_demanda
from datos import (ARCHIVO_DATOS, ARCHIVO_EPISODIOS, CONTEOS, VIVIENDAS, ENTRADAS, OBJETIVO,
                   cargar_cohorte, huella, separar_cohorte)
from modelo import (CATEGORICAS, NUMERICAS, RESULTADOS, construir_variables, predecir)

SEMILLA = 42
REFERENCIA = "Mediana histórica"


def crear_pipeline(regresor):
    """La imputación y la codificación se ajustan dentro de cada entrenamiento."""
    preparacion = ColumnTransformer([
        ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median")),
                                ("escalar", StandardScaler())]), NUMERICAS),
        ("categoricas", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                                  ("codificar", OneHotEncoder(handle_unknown="ignore",
                                                             sparse_output=False))]), CATEGORICAS),
    ])
    return Pipeline([("preparar", preparacion), ("regresor", regresor)])


def candidatos():
    # Las configuraciones se conservan fijas; no se ajustan observando la prueba.
    return [
        (REFERENCIA, "Mediana del objetivo en los datos de ajuste", DummyRegressor(strategy="median")),
        ("Regresión lineal", "Sin transformación del objetivo", LinearRegression()),
        ("Random Forest", "200 árboles; profundidad 8; hoja mínima 4",
         RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=4,
                               random_state=SEMILLA, n_jobs=1)),
        ("Gradient Boosting", "150 árboles; profundidad 2; tasa 0,04; pérdida Huber; hoja mínima 10",
         GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.04,
                                   min_samples_leaf=10, loss="huber", random_state=SEMILLA)),
    ]


def calcular_metricas(observado, estimado):
    observado, estimado = np.asarray(observado), np.asarray(estimado)
    return {"n": len(observado), "MAE": float(mean_absolute_error(observado, estimado)),
            "RMSE": float(root_mean_squared_error(observado, estimado)),
            "R2": float(r2_score(observado, estimado)) if len(observado) > 1 and np.ptp(observado) else None}


def guardar_graficos(carpeta, metricas, detalle):
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False})
    tabla = metricas.pivot(index="modelo", columns="etapa", values="MAE")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    tabla[["validacion", "prueba"]].rename(columns={"validacion": "Validación", "prueba": "Prueba"}).plot.barh(
        ax=ax, color=["#08796e", "#304c89"], edgecolor="none")
    ax.set(title="Error absoluto medio: menor es mejor", xlabel="Personas reportadas", ylabel="")
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(title=None)
    fig.savefig(carpeta / "comparacion_modelos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    maximo = max(float(detalle.observado.max()), float(detalle.estimado.max()), 1)
    ax.scatter(detalle.observado, detalle.estimado, color="#08796e", alpha=.6, s=20)
    ax.plot([0, maximo], [0, maximo], "--", color="#304c89", label="Estimación igual al registro")
    ax.set(xscale="symlog", yscale="symlog", xlim=(-.1, maximo * 1.15), ylim=(-.1, maximo * 1.15),
           title="Población registrada y estimada · prueba 2023–2024",
           xlabel="Personas reportadas", ylabel="Personas estimadas")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.15)
    fig.savefig(carpeta / "registrado_estimado.png", dpi=150)
    plt.close(fig)


def entrenar(ruta=ARCHIVO_DATOS, salida=RESULTADOS):
    ruta, salida = Path(ruta), Path(salida)
    cohorte = cargar_cohorte(ruta)
    grupos = separar_cohorte(cohorte)
    train, val, test = (grupos[n] for n in ("entrenamiento", "validacion", "prueba"))
    desarrollo = pd.concat([train, val], ignore_index=True)
    metricas, iniciales = [], {}
    for nombre, configuracion, regresor in candidatos():
        pipeline = crear_pipeline(regresor).fit(construir_variables(train), train[OBJETIVO])
        iniciales[nombre] = pipeline
        for etapa, registros in (("entrenamiento", train), ("validacion", val)):
            metricas.append({"modelo": nombre, "etapa": etapa, "configuracion": configuracion,
                             **calcular_metricas(registros[OBJETIVO], predecir(pipeline, registros))})

    # La referencia se compara con ML, pero no se presenta como un algoritmo aprendido.
    validacion_ml = [m for m in metricas if m["etapa"] == "validacion" and m["modelo"] != REFERENCIA]
    elegido = min(validacion_ml, key=lambda m: (m["MAE"], m["RMSE"], m["modelo"]))["modelo"]
    finales = {}
    for nombre, configuracion, _ in candidatos():
        final = clone(iniciales[nombre]).fit(construir_variables(desarrollo), desarrollo[OBJETIVO])
        finales[nombre] = final
        metricas.append({"modelo": nombre, "etapa": "prueba", "configuracion": configuracion,
                         **calcular_metricas(test[OBJETIVO], predecir(final, test))})

    tabla = pd.DataFrame(metricas)
    detalle = test[["id_sinpad", "fecha_emergencia", "ubigeo", "provincia", "distrito", "peligro",
                    "episodio", *CONTEOS]].copy()
    detalle["observado"] = test[OBJETIVO]
    detalle["estimado"] = predecir(finales[elegido], test)
    detalle["referencia"] = predecir(finales[REFERENCIA], test)
    detalle["error"] = detalle.estimado - detalle.observado
    detalle["error_absoluto"] = detalle.error.abs()
    provincias = []
    for provincia, registros in detalle.groupby("provincia"):
        provincias.append({"provincia": provincia,
                           **calcular_metricas(registros.observado, registros.estimado),
                           "MAE_referencia": float(mean_absolute_error(registros.observado, registros.referencia)),
                           "sesgo": float(registros.error.mean())})
    particiones = cohorte.groupby("particion", sort=False).agg(
        inicio=("fecha_emergencia", "min"), fin=("fecha_emergencia", "max"),
        registros=("id_sinpad", "size"), ceros=(OBJETIVO, lambda x: int(x.eq(0).sum())),
        positivos=(OBJETIVO, lambda x: int(x.gt(0).sum()))).reset_index()

    def resultado(nombre, etapa):
        return next(m for m in metricas if m["modelo"] == nombre and m["etapa"] == etapa)

    prueba_ml, prueba_ref = resultado(elegido, "prueba"), resultado(REFERENCIA, "prueba")
    distintos = int(detalle.estimado.round(8).nunique())
    meta = {
        "objetivo": OBJETIVO, "unidad": "personas reportadas por caso", "modelo": elegido,
        "configuracion": resultado(elegido, "validacion")["configuracion"], "semilla": SEMILLA,
        "n_cohorte": len(cohorte), "n_elegibles": len(desarrollo) + len(test),
        "n_excluidos": int(cohorte.particion.eq("excluido").sum()),
        "particiones": {nombre: len(registros) for nombre, registros in grupos.items()},
        "n_ceros_elegibles": int(cohorte.loc[cohorte.particion.ne("excluido"), OBJETIVO].eq(0).sum()),
        "predicciones_distintas": distintos, "predicciones_cero": int(detalle.estimado.eq(0).sum()),
        "supera_referencia_validacion": resultado(elegido, "validacion")["MAE"] < resultado(REFERENCIA, "validacion")["MAE"],
        "supera_referencia_prueba": prueba_ml["MAE"] < prueba_ref["MAE"],
        "prueba_modelo": prueba_ml, "prueba_referencia": prueba_ref,
        "validacion_modelo": resultado(elegido, "validacion"),
        "validacion_referencia": resultado(REFERENCIA, "validacion"),
        "mediana_ajuste": float(desarrollo[OBJETIVO].median()),
        "categorias": {c: sorted(desarrollo[c].dropna().unique().tolist()) for c in CATEGORICAS},
        "rangos": {c: [int(desarrollo[c].min()), int(desarrollo[c].max())] for c in VIVIENDAS},
        "combinaciones": desarrollo[CATEGORICAS].drop_duplicates().values.tolist(),
        "procedimiento": "Configuraciones fijas; elección del regresor por MAE de validación; "
                         "reajuste con entrenamiento y validación; comparación retrospectiva con el episodio 2023–2024.",
        "limites": "Las entradas contienen daños en viviendas ya observados. Se estima población reportada "
                   "por emergencia y se calculan requerimientos alimentarios normativos. No se mide demanda "
                   "alimentaria observada ni se pronostica la ocurrencia de El Niño. Evaluación retrospectiva; "
                   "no constituye una validación externa independiente.",
        "versiones": {"python": platform.python_version(), "pandas": pd.__version__,
                      "numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "huellas": {"dataset": huella(ruta), "episodios": huella(ARCHIVO_EPISODIOS)},
    }
    salida.mkdir(parents=True, exist_ok=True)
    joblib.dump(finales[elegido], salida / "modelo.joblib", compress=3)
    meta["huellas"]["modelo"] = huella(salida / "modelo.joblib")
    (salida / "resumen.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for nombre, registros in (("metricas.csv", tabla), ("predicciones_prueba.csv", detalle),
                              ("errores_por_provincia.csv", pd.DataFrame(provincias)),
                              ("particiones.csv", particiones), ("datos_estudio.csv", cohorte)):
        registros.to_csv(salida / nombre, index=False, encoding="utf-8-sig")
    guardar_graficos(salida, tabla, detalle)
    exportar_metricas_demanda(salida, detalle, meta)
    return tabla, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=ARCHIVO_DATOS)
    parser.add_argument("--salida", type=Path, default=RESULTADOS)
    args = parser.parse_args()
    tabla, meta = entrenar(args.datos, args.salida)
    print(tabla[["modelo", "etapa", "n", "MAE", "RMSE", "R2"]].round(4).to_string(index=False))
    print(f"Regresor elegido por validación: {meta['modelo']}.")
    print(f"Predicciones diferentes en prueba: {meta['predicciones_distintas']}.")
    print("Evaluación de población reportada y requerimientos alimentarios normativos.")
