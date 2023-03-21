# TFM - Generación de redes de asociación de genes
Este repositorio contiene el código realizado en el Trabajo de Fin de Máster (TFM).

## Uso

1. Instalar las dependencias del archivo `requirements.txt`.

2. Configurar los parámetros de opciones para el lanzamiento:
   - `dataset_name`: Nombre del dataset a procesar, debe almacenarse en la carpeta `./datasets`.
   - `kde_bins`: Número de bins con los que se calculará el KDE. Es fundamental para el cálculo de la distancia, ya que la distancia se calcula por diferencia de bins y categorías, la distancia máxima será bins * categorías.
   - `min_nodos_per_group`: Número mínimo de nodos dentro de un grupo de nodos en la red para tener en cuenta. Los grupos de nodos con menos nodos que este valor serán eliminados.
   - `max_nodos_network`: Número máximo de nodos que se tendrán en cuenta para el cálculo de la distancia del umbral. El nodo puede tener más nodos que los aquí establecidos, ya que si al calcular un umbral de distancia, desde 0 a esa distancia hay algunos nodos más, se tendrían en cuenta.
   - `complete_network`: Si es `True`, obtendremos, además del grafo de recubrimiento mínimo, con formato './outputs/{dataset_name}/graphs/graph_complete_{dataset_name}_umbral{round(self.draw_umbral, 2)}_nodes{self.max_nodos_network}_group{self.min_nodos_per_group}.html', los grafos completos generados, con el mismo formato de nombre, pero empezando por graph_complete

3. Ejecutar el archivo `main.py`.

Después de ejecutarlo, se generarán una serie de archivos en la carpeta `./outputs`. Cada dataset genera una carpeta con su nombre, donde se encuentran los grafos generados en `graphs`, y la lista de nodos en `validations`. Además, se guardará un archivo `distance_pair.json` que guarda el cálculo de la distancia, para poder realizar posteriores análisis sin la necesidad de recalcular todas las distancias, ya que es el cálculo que más tiempo consume. Los archivos se guardan con el nombre que les corresponda, más las variables de la ejecución.

## Pruebas
Para las pruebas, se ha dejado subido el dataset `Ovary_GSE12470` y el archivo `./outputs/Ovary_GSE12470/distance_pair.json` para tener un ejemplo con la distancia precalculada. Si se desea probar ese dataset calculando la distancia, eliminar el archivo y volver a lanzar.

## Datasets

Los datasets para las pruebas provienen de https://sbcb.inf.ufrgs.br/cumida.
