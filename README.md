# Proyecto 5: Trashnet Classification

Este proyecto implementa la clasificación de imágenes del dataset **Trashnet** utilizando una red neuronal convolucional pre-entrenada (**ResNet-18**) para extraer características, y una **Red Neuronal Competitiva (Winner-Takes-All)** para agrupar dichas características de forma no supervisada.


## Requisitos

Instala las dependencias necesarias. Se recomienda utilizar un entorno virtual (por ejemplo, con `conda` o `venv`):

```bash
pip install -r requirements.txt
```

## Dataset

El proyecto espera que el dataset Trashnet esté localizado en una carpeta llamada `dataset` en el mismo directorio que `main.py`.

1. Descarga el dataset de Kaggle: [Trashnet Dataset](https://www.kaggle.com/datasets/feyzazkefe/trashnet/data)
2. Descomprime el archivo descargado.
3. Asegúrate de que la estructura de carpetas quede así:

```
project5_trashnet/
│
├── dataset/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── main.py
├── ...
```

## Ejecución

Una vez configurado el dataset, simplemente corre el script principal:

```bash
python main.py
```

## Resultados

El sistema:
1. Cargará las imágenes.
2. Extraerá las características (vectores de 512 números) utilizando ResNet-18.
3. Entrenará la red competitiva utilizando la regla de aprendizaje de Kohonen.
4. Generará gráficas comparando los agrupamientos en la carpeta `results/`.
    - `distribution_comparison.png`: Compara la cantidad real de imágenes por clase vs las agrupadas por la red.
    - `cluster_compositions.png`: Muestra qué clases reales quedaron atrapadas en cada "Cluster" que la red competitiva descubrió.

