"""Estimación normativa de alimentos a partir de daños en viviendas de Piura."""
import pandas as pd
import streamlit as st

from demanda import calcular_requerimientos, FUENTE, DIAS_EVALUACION
from datos import (ARCHIVO_DATOS, ARCHIVO_EPISODIOS, CONTEOS, VIVIENDAS, ENTRADAS, OBJETIVO,
                   buscar_registros, cargar_cohorte, cargar_original)
from modelo import ETIQUETAS, MESES, RESULTADOS, cargar_modelo, estimar

st.set_page_config(page_title="Demanda alimentaria | Piura", page_icon="🌧️", layout="wide")


def formato(valor, decimales=2):
    if pd.isna(valor):
        return "Sin dato"
    return f"{valor:,.{decimales}f}".replace(",", " ").replace(".", ",")


@st.cache_data
def leer_datos(firma_datos, firma_episodios):
    return cargar_original(), cargar_cohorte()


@st.cache_resource
def leer_modelo(firma_resumen, firma_modelo, firma_datos, firma_episodios):
    return cargar_modelo()


def csv(registros):
    return registros.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig")


def nombre_caso(fila):
    return (f"SINPAD {fila.id_sinpad} · {fila.fecha_emergencia:%d/%m/%Y} · "
            f"{fila.distrito.title()} · {fila.peligro.title()}")


def buscar_y_elegir(base, clave):
    texto = st.text_input("Buscar código SINPAD, distrito, provincia, UBIGEO, peligro o fecha",
                          key=f"{clave}_texto", placeholder="Ejemplo: 186035 o Huarmaca")
    columnas = st.columns(3)
    provincia = columnas[0].selectbox("Provincia", [None, *sorted(base.provincia.unique())],
        format_func=lambda v: "Todas" if v is None else v.title(), key=f"{clave}_provincia")
    anio = columnas[1].selectbox("Año", [None, *sorted(base.anio.unique(), reverse=True)],
        format_func=lambda v: "Todos" if v is None else str(v), key=f"{clave}_anio")
    etiquetas = dict(zip(base.episodio, base.intervalo_episodio))
    episodio = columnas[2].selectbox("Episodio de El Niño costero", [None, *sorted(etiquetas, reverse=True)],
        format_func=lambda v: "Todos" if v is None else etiquetas[v], key=f"{clave}_episodio")
    encontrados = buscar_registros(base, texto, provincia, anio, episodio)
    st.caption(f"{len(encontrados)} registros encontrados de {len(base)} en esta sección.")
    if encontrados.empty:
        st.info("No hay registros que coincidan con la búsqueda y los filtros.")
        return None, encontrados
    opciones = {fila.id_sinpad: nombre_caso(fila) for fila in encontrados.itertuples()}
    llave = f"{clave}_caso"
    if st.session_state.get(llave) not in opciones:
        con_poblacion = encontrados.loc[encontrados.particion.eq("prueba") & encontrados[OBJETIVO].gt(0)]
        st.session_state[llave] = con_poblacion.iloc[0].id_sinpad if not con_poblacion.empty else next(iter(opciones))
    codigo = st.selectbox("Registro", list(opciones), format_func=opciones.get, key=llave)
    return encontrados.loc[encontrados.id_sinpad.eq(codigo)].iloc[0], encontrados


st.title("Demanda alimentaria en Piura")
st.write("Estimación de población a partir de daños en viviendas y cálculo de requerimientos alimentarios normativos.")
st.caption("Estudio de emergencias de Piura durante periodos de El Niño costero. "
           "La población procede de registros reales; las cantidades de alimentos se calculan con una ración de referencia.")
try:
    original, cohorte = leer_datos(ARCHIVO_DATOS.stat().st_mtime_ns, ARCHIVO_EPISODIOS.stat().st_mtime_ns)
    modelo, meta = leer_modelo((RESULTADOS / "resumen.json").stat().st_mtime_ns,
                              (RESULTADOS / "modelo.joblib").stat().st_mtime_ns,
                              ARCHIVO_DATOS.stat().st_mtime_ns, ARCHIVO_EPISODIOS.stat().st_mtime_ns)
    detalle = pd.read_csv(RESULTADOS / "predicciones_prueba.csv", dtype={"id_sinpad": str})
except (FileNotFoundError, ValueError, ImportError, KeyError) as exc:
    st.error(f"No se pudieron cargar los resultados: {exc}")
    st.code("python entrenar.py", language="bash")
    st.stop()

tarjetas = st.columns(3)
tarjetas[0].metric("Emergencias de Piura en periodos costeros", len(cohorte))
tarjetas[1].metric("Casos de prueba de 2023–2024", meta["particiones"]["prueba"])
tarjetas[2].metric("Modelo elegido por validación", meta["modelo"])
historico, estimacion, evaluacion = st.tabs(["1. Registros históricos", "2. Estimar alimentos", "3. Evaluación"])

with historico:
    st.subheader("Revisar los datos de una emergencia")
    caso, encontrados = buscar_y_elegir(cohorte, "historico")
    if caso is not None:
        st.caption(f"{caso.provincia.title()} · {caso.distrito.title()} · UBIGEO {caso.ubigeo} · "
                   f"Episodio: {caso.intervalo_episodio}")
        tarjetas = st.columns(4)
        etiquetas = {"afectados": "Personas afectadas", "damnificados": "Personas damnificadas", **ETIQUETAS}
        for tarjeta, campo in zip(tarjetas, CONTEOS):
            tarjeta.metric(etiquetas[campo], formato(caso[campo], 0))
        st.metric("Población reportada en el caso: afectados + damnificados", formato(caso[OBJETIVO], 0))
        st.caption("Estos conteos son los datos de referencia. Las personas de un caso pueden volver a figurar "
                   "en otra emergencia; no son personas únicas entre eventos.")
        columnas = ["id_sinpad", "fecha_emergencia", "provincia", "distrito", "peligro", *CONTEOS,
                    OBJETIVO, "intervalo_episodio", "particion"]
        st.dataframe(encontrados[columnas], hide_index=True, width="stretch")
        st.download_button("Descargar registros filtrados", csv(encontrados[columnas]), "registros_piura.csv", "text/csv")
    with st.expander("Fuentes y selección temporal"):
        st.write(f"El extracto original conserva {len(original)} registros. Se seleccionan {len(cohorte)} "
                 "que coinciden por mes con los intervalos de la tabla 3 de ENFEN. La coincidencia "
                 "temporal no demuestra que el fenómeno haya causado cada emergencia.")
        st.markdown("[Datos INDECI–SINPAD](https://www.datosabiertos.gob.pe/dataset/emergencias-y-da%C3%B1os-nivel-nacional-por-departamento) · "
                    "[Cronología ENFEN](https://enfen.imarpe.gob.pe/download/nota-tecnica-enfen-01-2024-definicion-operacional-de-los-eventos-el-nino-costero-y-la-nina-costera-en-el-peru/)")
        st.write("La cronología utilizada termina en 2024. Los registros de 2025–2026 no reciben una fase "
                 "climática por extrapolación. El peso y el costo de ayuda entregada no intervienen en el modelo.")

with estimacion:
    st.subheader("Calcular requerimientos de alimentos")
    st.write("El modelo recibe provincia, peligro, mes y daños en viviendas. Estima población reportada "
             "y aplica la ración logística del manual INDECI–PMA a los días seleccionados.")
    st.info("Escenario de cobertura total: se planifica una ración diaria por persona estimada. "
            "Los resultados son requerimientos normativos, no solicitudes observadas ni cantidades entregadas.")
    dias = st.number_input("Días de asistencia alimentaria", min_value=1, max_value=10, value=7, step=1)
    origen = st.radio("Datos de entrada", ["Caso real de prueba", "Escenario manual"], horizontal=True)
    entrada, caso_actual = None, None
    if origen == "Caso real de prueba":
        prueba = cohorte.loc[cohorte.particion.eq("prueba")]
        caso_actual, _ = buscar_y_elegir(prueba, "prueba")
        if caso_actual is not None:
            entrada = {c: caso_actual[c] for c in ENTRADAS}
    else:
        st.caption("Los daños del escenario deben conocerse o establecerse como supuesto. "
                   "Esta consulta no predice cuándo ocurrirá una emergencia.")
        with st.form("escenario"):
            columnas = st.columns(2)
            provincia = columnas[0].selectbox("Provincia del escenario", meta["categorias"]["provincia"], format_func=str.title)
            peligro = columnas[1].selectbox("Peligro del escenario", meta["categorias"]["peligro"], format_func=str.title)
            mes = st.selectbox("Mes", list(range(1, 13)), format_func=lambda m: MESES[m - 1])
            columnas = st.columns(2)
            viviendas = {c: columnas[i].number_input(ETIQUETAS[c], min_value=0, max_value=1_000_000_000,
                                                    value=10 if i == 0 else 2, step=1)
                         for i, c in enumerate(VIVIENDAS)}
            enviar = st.form_submit_button("Estimar alimentos", type="primary")
        if enviar:
            st.session_state["escenario_enviado"] = {"provincia": provincia, "peligro": peligro, "mes": mes, **viviendas}
        entrada = st.session_state.get("escenario_enviado")
    if entrada is not None:
        try:
            personas, avisos = estimar(modelo, meta, entrada)
            requerimientos = calcular_requerimientos(personas, int(dias))
            st.metric("Población estimada por el modelo", formato(personas, 1))
            st.caption(f"La planificación redondea hacia arriba a {int(requerimientos.personas_planificadas.iloc[0])} "
                       "personas. No se suman kilogramos y litros como si fueran una misma unidad.")
            for aviso in avisos:
                st.warning(aviso)
            mostrar = requerimientos[["recurso", "unidad", "cantidad"]].copy()
            if caso_actual is not None:
                real = float(caso_actual[OBJETIVO])
                referencia = calcular_requerimientos(real, int(dias))
                tarjetas = st.columns(2)
                tarjetas[0].metric("Población registrada en el caso", formato(real, 0))
                tarjetas[1].metric("Error absoluto de población", formato(abs(personas - real), 1))
                mostrar["referencia_normativa"] = referencia.cantidad
                mostrar["error_absoluto_normativo"] = (mostrar.cantidad - mostrar.referencia_normativa).abs()
                st.caption("La referencia alimentaria se calcula con la población histórica; "
                           "no procede de una medición independiente de necesidades alimentarias.")
            st.dataframe(mostrar, hide_index=True, width="stretch")
            salida = requerimientos.assign(poblacion_estimada=personas, modelo=meta["modelo"],
                                            origen_entrada=origen, fuente_racion=FUENTE)
            if caso_actual is not None:
                salida = salida.assign(id_sinpad=caso_actual.id_sinpad,
                                        poblacion_registrada=float(caso_actual[OBJETIVO]),
                                        referencia_normativa=referencia.cantidad)
            st.download_button("Descargar requerimientos", csv(salida), "requerimientos_alimentarios.csv", "text/csv")
        except ValueError as exc:
            st.error(str(exc))
    with st.expander("Fórmula y raciones utilizadas"):
        st.markdown(f"[Manual INDECI–PMA de 2017]({FUENTE}), cuadro 7 y procedimiento de requerimiento alimentario.")
        st.write("Cantidad = población planificada × días × ración por persona y día. "
                 "El ejemplo de alimentos no incorpora dietas especiales ni demuestra que toda persona afectada "
                 "necesite recibir la ración completa; la cobertura total es un supuesto de este escenario.")
        st.dataframe(calcular_requerimientos(1, 1)[["recurso", "unidad", "racion_persona_dia"]], hide_index=True)

with evaluacion:
    st.subheader("Comprobar el modelo con registros posteriores")
    st.write("La evaluación contrasta población registrada y estimada. Los afectados y damnificados "
             "sirven para comprobar la respuesta; no se entregan como entradas del modelo.")
    tarjetas = st.columns(3)
    error = meta["prueba_modelo"]["MAE"]
    error_base = meta["prueba_referencia"]["MAE"]
    tarjetas[0].metric("MAE del modelo · personas", formato(error, 1))
    tarjetas[1].metric("MAE de la mediana · personas", formato(error_base, 1))
    tarjetas[2].metric("R² en prueba", formato(meta["prueba_modelo"]["R2"], 3))
    if meta["supera_referencia_prueba"]:
        st.success(f"El error absoluto medio fue {100 * (1 - error / error_base):.1f}% menor que el de la mediana "
                   "en esta prueba retrospectiva. Esta reducción no es un porcentaje de exactitud.")
    else:
        st.warning("El modelo no mejoró a la mediana histórica en esta prueba.")
    st.caption(f"El modelo produjo {meta['predicciones_distintas']} valores distintos en "
               f"{meta['particiones']['prueba']} casos. Las métricas no garantizan rendimiento futuro ni uso operativo.")
    metricas = pd.read_csv(RESULTADOS / "metricas.csv")
    st.dataframe(metricas.loc[metricas.etapa.eq("prueba"), ["modelo", "n", "MAE", "RMSE", "R2"]].round(3),
                 hide_index=True, width="stretch")
    columnas = st.columns(2)
    columnas[0].image(str(RESULTADOS / "comparacion_modelos.png"), width="stretch")
    columnas[1].image(str(RESULTADOS / "registrado_estimado.png"), width="stretch")
    with st.expander("Particiones y configuraciones"):
        st.dataframe(pd.read_csv(RESULTADOS / "particiones.csv"), hide_index=True, width="stretch")
        st.dataframe(metricas.round(3), hide_index=True, width="stretch")
        st.caption("Entrenamiento hasta 2016; validación 2017–2022; prueba del episodio 2023–2024. "
                   "Las configuraciones son fijas. El modelo se elige por MAE de validación y se reajusta "
                   "con entrenamiento y validación, sin dividir episodios entre particiones.")
    with st.expander("Error propagado a los requerimientos alimentarios"):
        normativas = pd.read_csv(RESULTADOS / "metricas_requerimientos.csv")
        st.write(f"Escenario fijo de {DIAS_EVALUACION} días. Las diferencias se calculan contra raciones "
                 "derivadas de la población registrada. No son errores medidos contra solicitudes alimentarias reales.")
        st.dataframe(normativas.round(3), hide_index=True, width="stretch")
    with st.expander("Errores por provincia y límites de interpretación"):
        st.dataframe(pd.read_csv(RESULTADOS / "errores_por_provincia.csv").round(3), hide_index=True, width="stretch")
        st.write(meta["limites"])
        st.write("No se incorporaron series de lluvia, caudal o temperatura del mar. "
                 "La evaluación es retrospectiva y no es una validación externa independiente. "
                 "Las dietas especiales, el stock, la capacidad logística y el acceso real a alimentos no están modelados.")
    st.download_button("Descargar métricas de población", csv(metricas), "metricas_poblacion.csv", "text/csv")
    st.download_button("Descargar comparaciones de prueba", csv(detalle), "comparaciones_prueba.csv", "text/csv")


