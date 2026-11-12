# Demanda alimentaria en Piura

Aplicación académica para estimar requerimientos alimentarios normativos en emergencias de Piura durante periodos de El Niño costero. Se combina un modelo de población reportada con las raciones logísticas del manual INDECI–PMA de 2017.

El modelo recibe **provincia, peligro, mes y viviendas afectadas y destruidas**. Estima la suma de personas afectadas y damnificadas reportadas por emergencia. Esa población se redondea hacia arriba y se convierte en cantidades de alimentos para un escenario de cobertura total y entre 1 y 10 días.

Las cantidades de alimentos son calculadas, no observaciones de solicitudes ni entregas. El sistema utiliza daños en viviendas ya observados o planteados como escenario; no pronostica cuándo ocurrirá El Niño ni anticipa emergencias sin conocer esos daños. El alcance es la planificación normativa, no una validación de demanda alimentaria observada.

## Ejecución

Extraer la carpeta `proyecto_machine_piura`. Se requiere Python 3.12. En Windows, abrir `iniciar.bat`, o ejecutar desde esa carpeta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si ya existe un entorno compatible, omitir su creación. En Linux y macOS se utilizan `python3.12 -m venv .venv` y `.venv/bin/python` como ejecutable. La aplicación incluye el modelo entrenado y no necesita Internet para las consultas. Mantener la terminal abierta; `Ctrl+C` detiene el servidor.

## Secciones

| Sección | Función |
|---|---|
| Registros históricos | Buscar emergencias reales por código, ubicación, peligro o fecha; filtrar por provincia, año y episodio; revisar daños y población reportada. |
| Estimar alimentos | Consultar un caso de prueba o un escenario de daños en viviendas; estimar población y requerimientos de cinco grupos alimentarios; comparar con la referencia normativa de la población histórica. |
| Evaluación | Comparar los modelos con una mediana, revisar errores y exportar resultados. Separar evaluación empírica de población y cálculo derivado de alimentos. |

Ejemplo: el código **185838**, Ayabaca, 27 de diciembre de 2023, registra 10 viviendas afectadas, una destruida y 23 personas entre afectadas y damnificadas. El modelo recibe los datos de viviendas, provincia, peligro y mes; las 23 personas se reservan para la comparación y no se introducen como predictor.

## Datos y método

El CSV y el Excel originales conservan sus bytes: 2 326 emergencias de Piura, 28 columnas y cinco peligros hidrometeorológicos entre 2003 y 2025. El programa deriva la población reportada sumando los dos conteos de personas, sin modificar los archivos. No se trata de personas únicas entre eventos.

El cruce mensual con la tabla 3 de ENFEN identifica 899 registros. La coincidencia temporal no establece causalidad. La cronología empleada termina en 2024 y no clasifica 2025–2026 por extrapolación.

| Partición | Periodo | Casos |
|---|---|---:|
| Entrenamiento | Hasta 2016 | 136 |
| Validación | 2017–2022 | 368 |
| Prueba | Episodio 2023–2024 | 395 |

Los 76 registros del episodio en enero–febrero de 2024 tienen conteos de personas disponibles y se incluyen en prueba. La falta de cobertura de ayuda entregada no los excluye de un modelo cuyo objetivo es población. No se divide un episodio entre particiones elegibles. Se mantienen los ceros de población como reportados; su completitud sigue siendo una limitación documental. No se utiliza el peso ni el costo de ayuda para entrenar, filtrar por la respuesta o calcular alimentos.

El preprocesamiento utiliza imputación numérica, escalado, OneHotEncoder y representación cíclica del mes, ajustados dentro del pipeline. Se comparan tres familias con una configuración fija por familia; se elige el regresor por MAE de validación y se reajusta con entrenamiento y validación antes de evaluar la prueba.

| Modelo | MAE validación | MAE prueba | RMSE prueba | R² prueba |
|---|---:|---:|---:|---:|
| Mediana histórica | 1 043,28 | 791,92 | 2 112,06 | −0,144 |
| Regresión lineal | 790,82 | 794,68 | 1 885,22 | 0,089 |
| Random Forest | 728,49 | 1 014,03 | 2 328,34 | −0,390 |
| Gradient Boosting | 595,86 | 540,54 | 1 271,04 | 0,586 |

MAE y RMSE se expresan en personas reportadas. Gradient Boosting reduce el MAE de prueba un 31,7 % frente a la mediana y produce 221 valores distintos. Esta reducción no es un porcentaje de exactitud. El error sigue siendo importante y no demuestra fiabilidad operativa ni rendimiento futuro.

La evaluación es retrospectiva; 2023 ya se había revisado durante el desarrollo, aunque para otro objetivo. No es validación externa independiente. No se ajustan parámetros mirando las métricas de prueba del nuevo objetivo.

## Cálculo alimentario

`demanda.py` aplica **cantidad = población planificada × días × ración por persona y día**. Se emplea el cuadro 7 del manual INDECI–PMA de 2017:

| Grupo | Cantidad por persona y día | Unidad |
|---|---:|---|
| Cereales | 0,300 | kg |
| Menestras | 0,150 | kg |
| Azúcar | 0,040 | kg |
| Aceite vegetal | 0,040 | litros |
| Conserva de origen animal | 0,060 | kg |

No se suman litros con kilogramos ni se supone una densidad del aceite para calcular un peso total. La conserva corresponde a la ración logística, no a un cálculo nutricional individualizado. Las dietas especiales, existencias, presupuesto, accesibilidad y necesidad efectiva de asistencia no están modelados. La cobertura completa es un supuesto explícito de planificación, no una observación de inseguridad alimentaria.

Las métricas de alimentos comparan cantidades derivadas de la población estimada con cantidades derivadas de la población registrada. No constituyen una segunda validación independiente ni una medición contra necesidades alimentarias observadas.

## Reproducir y comprobar

```powershell
.\.venv\Scripts\python.exe entrenar.py
.\.venv\Scripts\python.exe -m unittest discover -s pruebas -v
.\.venv\Scripts\python.exe -m pip check
```

`Proyecto_Machine_Piura.ipynb` reproduce el estudio en Jupyter o Colab. En Colab se carga `proyecto_machine_piura.zip` cuando se solicita. Las salidas incluidas se ejecutaron localmente; no se atribuye una comprobación de la sesión web de Colab.

Las pruebas comprueban territorio, fechas, separación de episodios, ausencia de población en las entradas, inferencia, fórmulas, unidades, interfaz, gráficos y descargas. Son controles de software y consistencia, no validación operativa.

## Estructura

- `app.py`: interfaz de tres secciones.
- `datos.py`: carga, población de referencia, selección y búsqueda.
- `modelo.py`: variables y consulta del regresor.
- `demanda.py`: raciones y cálculo de cantidades normativas.
- `entrenar.py`: comparación, métricas y exportaciones.
- `datos/`: archivos originales y cronología costera.
- `resultados/`: modelo, comparaciones, métricas de población y requerimientos derivados.
- `pruebas/`: comprobaciones automatizadas.
- `GUIA_PROYECTO.md`: explicación de los bloques de código.
- `AUDITORIA_DATOS.md`: diagnóstico de ceros, cobertura y unidades de la ayuda entregada.
- `AJUSTES_INFORME.md`: redacción coherente con este alcance.
- `FUENTES.md`: procedencia y criterios de uso.
