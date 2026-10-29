# Detección de melanoma

Proyecto de investigación para entrenar y evaluar un clasificador de lesiones cutáneas a partir de imágenes dermatoscópicas. El repositorio incluye un pipeline de datos y entrenamiento, evaluación del modelo y visualización de activaciones mediante Grad-CAM++.

El software es experimental y no está destinado al diagnóstico ni a la toma de decisiones clínicas.

## Contenido

- `run_pipeline.py`: prepara los datos, entrena el modelo y ejecuta la evaluación y el análisis XAI.
- `evaluate_trained_model.py`: evalúa un checkpoint existente sin volver a entrenar.
- `run_xai_only.py`: genera visualizaciones Grad-CAM++ a partir de un modelo y metadatos existentes.
- `src/data_pipeline.py`: carga y prepara los datos, crea los conjuntos de datos y los cargadores.
- `src/model.py`: definición y entrenamiento del clasificador.
- `src/evaluation.py`: métricas, calibración y análisis estadístico.
- `src/xai_gradcam.py`: generación y análisis de mapas de activación.
- `melanoma_detection.ipynb`: notebook de exploración del proyecto.
- `tests/`: pruebas automatizadas.

Los directorios `data/`, `checkpoints/` y `results/` contienen datos o artefactos generados y están excluidos del control de versiones.

## Requisitos

- Python 3.10 o posterior.
- Dependencias de Python indicadas en [`requirements.txt`](requirements.txt).
- Para entrenar con un rendimiento razonable, se recomienda una GPU compatible con PyTorch. También se puede ejecutar en CPU, con un coste de tiempo mayor.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Windows, activa el entorno con `.venv\Scripts\activate`.

## Datos

El pipeline está configurado para trabajar con ISIC 2020. Puede utilizar un conjunto de imágenes y metadatos locales o intentar obtener imágenes mediante la API de ISIC Archive. La configuración y las rutas se definen en `CONFIG` dentro de `run_pipeline.py`.

Para hacer una partición estricta por paciente, el pipeline requiere identificadores de paciente fiables. Si se usa el CSV de etiquetas de ISIC 2020, configúralo en `isic2020_ground_truth_csv`; las imágenes locales se indican en `isic2020_images_dir`. Por defecto, el entrenamiento se detiene si no puede verificar el agrupamiento por paciente. No desactives esta comprobación para resultados que se vayan a presentar como evaluación independiente.

Los datos de ISIC están sujetos a sus propias condiciones de acceso y uso. Consulta la fuente del conjunto antes de descargarlo o redistribuirlo.

## Ejecución

Inicia el pipeline de entrenamiento y evaluación con:

```bash
python run_pipeline.py
```

El script guarda checkpoints en `checkpoints/main/` y resultados en `results/`. Los nombres y la configuración de salida se pueden consultar en `run_pipeline.py`.

Para evaluar un modelo ya entrenado, asegúrate de que el checkpoint y el `data/metadata.csv` correspondiente estén disponibles y ejecuta:

```bash
python evaluate_trained_model.py
```

Para generar mapas Grad-CAM++ sin volver a entrenar:

```bash
python run_xai_only.py
```

El notebook puede abrirse con Jupyter:

```bash
jupyter notebook melanoma_detection.ipynb
```

## Modelo y evaluación

La configuración actual de `run_pipeline.py` usa EfficientNet-B4 con pesos preentrenados en ImageNet y entradas de 380 × 380 píxeles. El entrenamiento incluye ponderación de clases, pérdida focal, aumentación, pesos EMA y precisión mixta cuando el dispositivo lo permite. Los detalles configurables están en el diccionario `CONFIG` y en los módulos de `src/`.

La evaluación incluye métricas de clasificación, selección del umbral con el conjunto de validación, calibración de temperatura y estimaciones bootstrap de intervalos de confianza. Grad-CAM++ proporciona visualizaciones de las regiones que contribuyen a la salida del modelo; estas visualizaciones no validan por sí mismas la explicación ni el rendimiento clínico.

Los resultados dependen del conjunto de datos, la partición, el checkpoint y la configuración usados. No se incluyen aquí cifras de rendimiento como resultados de referencia, porque deben acompañarse de la procedencia de los datos y de la configuración exacta de cada ejecución.

## Uso responsable y limitaciones

Este proyecto es únicamente para investigación y desarrollo. No es un producto sanitario ni ha sido validado para uso clínico. Antes de considerar cualquier aplicación clínica se necesitarían, entre otros aspectos, validación externa independiente, evaluación prospectiva, análisis de rendimiento por subgrupos y revisión regulatoria aplicable.

Una partición por paciente ayuda a reducir la fuga de información entre entrenamiento y evaluación, pero no sustituye la validación externa. Los mapas de activación son herramientas exploratorias y no deben interpretarse como evidencia de causalidad o razonamiento clínico.
