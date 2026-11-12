"""Carga, validación territorial y selección temporal de las emergencias de Piura."""
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "datos"
ARCHIVO_DATOS = CARPETA_DATOS / "dataset_piura.csv"
ARCHIVO_EPISODIOS = CARPETA_DATOS / "episodios_enfen.csv"
OBJETIVO = "poblacion_reportada"
CONTEOS = ["afectados", "damnificados", "viviendas_afectadas", "viviendas_destruidas"]
VIVIENDAS = ["viviendas_afectadas", "viviendas_destruidas"]
ENTRADAS = ["provincia", "peligro", "mes", *VIVIENDAS]
FECHA_LIMITE = pd.Timestamp("2024-02-29")


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_original(ruta=ARCHIVO_DATOS):
    """Conserva los valores de origen y comprueba identificadores, fechas y conteos."""
    ruta = Path(ruta)
    tipos = {"id_sinpad": str, "ubigeo": str}
    if ruta.suffix.lower() == ".xlsx":
        base = pd.read_excel(ruta, sheet_name="Datos", dtype=tipos)
    else:
        base = pd.read_csv(ruta, dtype=tipos)
    obligatorias = {*ENTRADAS, *CONTEOS, "id_sinpad", "ubigeo", "departamento",
                    "distrito", "anio", "fecha_emergencia", "revisar_cobertura_ayuda"}
    if faltantes := obligatorias.difference(base.columns):
        raise ValueError("Columnas faltantes: " + ", ".join(sorted(faltantes)))
    if base.empty:
        raise ValueError("El archivo de emergencias está vacío.")
    for campo in ("id_sinpad", "ubigeo", "departamento", "provincia", "distrito", "peligro"):
        if base[campo].isna().any():
            raise ValueError(f"Hay valores vacíos en {campo}.")
        base[campo] = base[campo].str.strip()
        if base[campo].eq("").any():
            raise ValueError(f"Hay valores vacíos en {campo}.")
    if not base.id_sinpad.str.fullmatch(r"[1-9][0-9]*").all() or base.id_sinpad.duplicated().any():
        raise ValueError("Los códigos SINPAD deben ser válidos y únicos.")
    if not (base.departamento.eq("PIURA").all()
            and base.ubigeo.str.fullmatch(r"20[0-9]{4}").all()):
        raise ValueError("El estudio admite exclusivamente registros del departamento de Piura.")
    base["fecha_emergencia"] = pd.to_datetime(base.fecha_emergencia, errors="raise")
    if base.fecha_emergencia.isna().any():
        raise ValueError("Hay fechas de emergencia vacías.")
    for campo in [*CONTEOS, "anio", "mes", "revisar_cobertura_ayuda"]:
        base[campo] = pd.to_numeric(base[campo], errors="raise")
        if np.isinf(base[campo]).any() or base[campo].lt(0).any():
            raise ValueError(f"Hay valores negativos o infinitos en {campo}.")
    for campo in CONTEOS:
        if not base[campo].dropna().mod(1).eq(0).all():
            raise ValueError(f"{campo} debe contener conteos enteros.")
    if not (base.anio.eq(base.fecha_emergencia.dt.year)
            & base.mes.eq(base.fecha_emergencia.dt.month)).all():
        raise ValueError("El año o el mes no corresponde a la fecha del registro.")
    if not base.revisar_cobertura_ayuda.isin([0, 1]).all():
        raise ValueError("La marca de revisión de cobertura debe ser 0 o 1.")
    # La suma corresponde a personas reportadas por caso, no a individuos únicos entre eventos.
    base[OBJETIVO] = base.afectados + base.damnificados
    return base.sort_values(["fecha_emergencia", "id_sinpad"]).reset_index(drop=True)


def cargar_episodios(ruta=ARCHIVO_EPISODIOS):
    episodios = pd.read_csv(ruta)
    campos = {"episodio_nino_costero", "inicio_mes", "fin_mes", "duracion_meses"}
    if not campos.issubset(episodios.columns) or episodios.empty:
        raise ValueError("La cronología de El Niño costero no tiene la estructura requerida.")
    meses_usados = set()
    for fila in episodios.itertuples(index=False):
        inicio, fin = pd.Period(fila.inicio_mes, "M"), pd.Period(fila.fin_mes, "M")
        meses = set(pd.period_range(inicio, fin, freq="M"))
        if fin < inicio or len(meses) != fila.duracion_meses or meses_usados.intersection(meses):
            raise ValueError("La cronología contiene intervalos inválidos o superpuestos.")
        meses_usados.update(meses)
    return episodios


def seleccionar_cohorte(original, episodios):
    """La coincidencia mensual contextualiza el registro; no demuestra causalidad."""
    base = original.copy()
    base["episodio"] = ""
    base["intervalo_episodio"] = ""
    meses = base.fecha_emergencia.dt.to_period("M")
    for fila in episodios.itertuples(index=False):
        coincide = meses.between(pd.Period(fila.inicio_mes, "M"), pd.Period(fila.fin_mes, "M"))
        base.loc[coincide, "episodio"] = fila.episodio_nino_costero
        base.loc[coincide, "intervalo_episodio"] = f"{fila.inicio_mes} a {fila.fin_mes}"
    base = base.loc[base.episodio.ne("")].copy()
    # El límite mantiene una prueba retrospectiva fija aunque se amplíe el archivo fuente.
    elegible = (base[OBJETIVO].notna()
                & base.fecha_emergencia.le(FECHA_LIMITE))
    base["particion"] = np.select(
        [~elegible, base.anio.le(2016), base.anio.le(2022)],
        ["excluido", "entrenamiento", "validacion"], default="prueba")
    base["motivo_exclusion"] = ""
    base.loc[base[OBJETIVO].isna(), "motivo_exclusion"] = "Sin conteo de población completo"
    base.loc[base.fecha_emergencia.gt(FECHA_LIMITE), "motivo_exclusion"] = "Fuera del periodo de evaluación"
    return base.reset_index(drop=True)


def cargar_cohorte(ruta=ARCHIVO_DATOS):
    return seleccionar_cohorte(cargar_original(ruta), cargar_episodios())


def separar_cohorte(cohorte):
    grupos = {nombre: cohorte.loc[cohorte.particion.eq(nombre)].copy()
              for nombre in ("entrenamiento", "validacion", "prueba")}
    if any(grupo.empty for grupo in grupos.values()):
        raise ValueError("No hay observaciones suficientes para las tres particiones temporales.")
    train, val, test = grupos.values()
    if not (train.fecha_emergencia.max() < val.fecha_emergencia.min()
            and val.fecha_emergencia.max() < test.fecha_emergencia.min()):
        raise ValueError("Las particiones no respetan el orden temporal.")
    elegibles = cohorte.loc[cohorte.particion.ne("excluido")]
    if elegibles.groupby("episodio").particion.nunique().gt(1).any():
        raise ValueError("Un episodio costero cruza las particiones elegibles.")
    return grupos


def normalizar_texto(texto):
    return "".join(letra for letra in unicodedata.normalize("NFKD", str(texto)).casefold()
                   if not unicodedata.combining(letra))


def buscar_registros(base, texto="", provincia=None, anio=None, episodio=None):
    resultado = base.copy()
    for campo, valor in (("provincia", provincia), ("anio", anio), ("episodio", episodio)):
        if valor is not None:
            resultado = resultado.loc[resultado[campo].eq(valor)]
    campos = ["id_sinpad", "ubigeo", "fecha_emergencia", "provincia", "distrito", "peligro"]
    indice = resultado[campos].fillna("").astype(str).agg(" ".join, axis=1).map(normalizar_texto)
    for palabra in normalizar_texto(texto).split():
        resultado = resultado.loc[indice.loc[resultado.index].str.contains(palabra, regex=False)]
    return resultado.sort_values(["fecha_emergencia", "id_sinpad"], ascending=False)
