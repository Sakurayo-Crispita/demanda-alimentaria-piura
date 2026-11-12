# Fuentes y trazabilidad

## Datos utilizados

**Instituto Nacional de Defensa Civil.** (s. f.). *Emergencias y daños a nivel nacional por departamento* [Conjunto de datos]. Plataforma Nacional de Datos Abiertos.

https://www.datosabiertos.gob.pe/dataset/emergencias-y-da%C3%B1os-nivel-nacional-por-departamento

Se conserva el extracto original recibido en el proyecto: 2 326 emergencias de Piura y 28 columnas. No se presenta como una nueva descarga completa o actualizada del portal. El peso y el costo de entregas se mantienen para trazabilidad, pero no intervienen en el modelo de población ni en el cálculo alimentario. La población de referencia se deriva exclusivamente de `afectados + damnificados` y conserva la naturaleza de conteo reportado por caso.

**Comisión Multisectorial ENFEN.** (2024). *Definición operacional de los eventos El Niño Costero y La Niña Costera en el Perú* (Nota Técnica ENFEN 01-2024).

https://enfen.imarpe.gob.pe/download/nota-tecnica-enfen-01-2024-definicion-operacional-de-los-eventos-el-nino-costero-y-la-nina-costera-en-el-peru/

Se utiliza la tabla 3, página 5, para seleccionar coincidencias mensuales con El Niño costero. El CSV conserva los intervalos pertinentes de esa tabla. No se atribuye causalidad individual, no se confunde con el ENOS global y no se clasifica 2025–2026 con una cronología que termina en 2024.

**Instituto Nacional de Defensa Civil y Programa Mundial de Alimentos.** (2017). *Manual para la asistencia alimentaria en emergencias en el Perú*.

https://bvpad.indeci.gob.pe/doc/folletos/2017/Manual%20para%20la%20AAE%20Peru.pdf

El cuadro 7, página impresa 10, define la ración logística utilizada. La sección 6.1.2, páginas impresas 24–26, explica el cálculo con población, días y ración. Se conserva el límite de 10 días como marco del escenario de raciones crudas de esa referencia histórica. No se presenta el documento como comprobación de la normativa actualmente vigente ni como una dieta individualizada.

Las cantidades son requerimientos normativos bajo cobertura total. No se incorporan etiquetas inventadas de solicitudes, necesidades de campo o entregas. No se utiliza un total de kg que sume aceite en litros.

## Auditoría del objetivo anterior

Los documentos de SINPAD y el reporte COEN-INDECI de Piura de marzo de 2024 permiten distinguir requerimientos, entregas y sus registros. La evidencia y los enlaces se conservan en `AUDITORIA_DATOS.md`.

La matriz DANA 2017 de IFRC se descargó y revisó: contiene daños y población, no cantidades requeridas de los grupos alimentarios. El catálogo DTM ronda 3 de OIM identifica una encuesta de desplazados de Catacaos y Cura Mori, pero no se logró descargar sus recursos; su contenido no se presenta como verificado. La tesis de Corcuera García (2020), de UDEP, utiliza población estimada para planificar almacenes; sus cantidades calculadas no se trataron como necesidades históricas observadas.

- IFRC: https://data.humdata.org/dataset/peru-floods-dana-matrix-glide-number-fl2017-000014
- OIM: https://data.humdata.org/dataset/peru-dtmr3
- Corcuera García, Á. (2020). *Propuesta de localización de almacenes de ayuda humanitaria para una atención eficiente de damnificados ante el Fenómeno El Niño en la región Piura* [Tesis de grado, Universidad de Piura]. https://hdl.handle.net/11042/4601

Estas fuentes complementarias no se mezclaron con el conjunto de entrenamiento. La evaluación empírica del proyecto corresponde a población reportada; las métricas alimentarias son una propagación del error mediante reglas explícitas.
