import collections
import os
import re
import time
import csv
import logging
import pandas as pd
from pyvis.network import Network
import numpy as np
from scipy import stats
import json

dataset_name = 'Ovary_GSE12470'
dataset_path = f'./datasets/' + dataset_name + '.csv'
offset_print = 50

class Main:
    def __init__(self):
        global dataset_path
        global dataset_name

        # Development options
        self.kde_bins = 100
        self.min_nodos_per_group = 3
        self.max_nodos_network = 100
        self.complete_network = False


    def run(self):
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

        print('INICIO EJECUCION')

        self.inicio = time.time()
        self.inicio_total = time.time()

        self.distance_memory = collections.defaultdict(dict)
        # Variable para almacenar los pares de nodos segun su distancia
        self.pairs_by_distance = {}
        self.pairs_by_distance[0] = []

        os.makedirs(f'./outputs/{dataset_name}/graphs', exist_ok=True)
        os.makedirs(f'./outputs/{dataset_name}/validation', exist_ok=True)

        with open(dataset_path, newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)
            if("," in row1):
                dataset = pd.read_csv(dataset_path, sep = ',')
            elif(";" in row1[0]):
                dataset = pd.read_csv(dataset_path, sep = ';')
            else:
                dataset = pd.read_csv(dataset_path)

        dataset_c = dataset.copy()
        del dataset_c['samples']
        
        self.fin = time.time()
        print(f'LECTURA DATOS: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print))
        self.inicio = time.time()

        self.columns = dataset_c.columns[1:]
        self.category_column_index = dataset_c.columns[0]
        self.categories = dataset_c[dataset_c.columns[0]].unique()

        normalized_dataset = self.normalize_dataset(dataset_c)
        self.fin = time.time()

        print(f'NORMALIZACION DATOS: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print))

        self.inicio = time.time()
        data_by_vars_categories = self.split_dataset_vars_categories(normalized_dataset)
        self.fin = time.time()

        print(f'DATOS POR CATEGORIA: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print))

        self.inicio = time.time()
        kde = self.get_kde(data_by_vars_categories)
        self.fin = time.time()

        print(f'CALCULO KDE: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print))

        self.inicio = time.time()

        self.own_representation = self.transform_kde_own_representation(kde)
        self.fin = time.time()

        print(f'VARIABLES POR CATEGORIA: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print) )

        self.inicio = time.time()

        self.get_distance_matrix()

        self.fin = time.time()
        print(f'CALCULO MATRIZ DISTANCIA: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print) )
        self.inicio = time.time()
        self.inicio = time.time()
        self.normalize_distance_pairs()
        network = self.get_network(0)
        self.fin = time.time()
        print(f'CALCULO GRAFO: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print) )
        print('Exportando el fichero de validacion')
        self.export_validation_file(network)
        self.fin_total = time.time()
        self.fin = time.time()
        print(f'PROCESO TOTAL NETWORK: {self.fin-self.inicio:.3f} sec.'.rjust(offset_print) )
        print('-------------------------------')
        print(f'TOTAL EJECUCION: {self.fin_total-self.inicio_total:.3f} sec.'.rjust(offset_print) )
   
    def get_distance_matrix(self):
        """
        Calcula la matriz de distancia entre variables del dataset.
        
        :param own_representation:
        Diccionario en el que cada key es una variable, y el value el nuevo vector de representacion de la variable, con el formato.
        {
            variable1 : [[cat1,cat2], [cat1,cat2], [cat1,cat2], [cat2,cat1], [cat2,cat1], [cat1,cat2],.... ],
        }
        """
        """
        res = []
        for var1 in own_representation:
            row = []
            for var2 in own_representation:
                row.append(self.get_distance_pair_vars(var1, var2, own_representation[var1], own_representation[var2]))
            res.append(row)
        """
        if os.path.exists(f'./outputs/{dataset_name}/distance_pair.json'):
            with open(f'./outputs/{dataset_name}/distance_pair.json', "r") as file:
                self.pairs_by_distance = json.load(file)
                self.pairs_by_distance = {int(key): value for key, value in self.pairs_by_distance.items()}
                return
            
        keys = list(self.own_representation.keys())
        # esto son la maxima distancia posible, por lo que si pasa por encima no guardamos datos.
        max_distance_save = int((self.kde_bins*len(self.own_representation[keys[0]][0]))*0.25)
        for i, var1 in enumerate(keys):
            for j, var2 in enumerate(keys[i+1:]):
                var1_data = self.own_representation[var1]
                var2_data = self.own_representation[var2]
                self.get_distance_pair_vars(var1_data, var2_data, var1, var2, max_distance_save)
 

        with open(f'./outputs/{dataset_name}/distance_pair.json', "w") as file:
            json.dump(self.pairs_by_distance, file)


    def split_dataset_vars_categories(self, normalized_data):
        """
        divide el dataset en subconjuntos por categoría y crea un diccionario que contiene los valores de cada variable para cada categoría.
        :param normalized_data:
            Dataset de pandas normalizado.

        :dataset_category
        Objeto intermedio en forma de diccionario que organiza las muestras del dataset según su categoría.
        La categoría se identifica mediante una clave en el diccionario y se le asigna un valor que es un subconjunto de las muestras del dataset pertenecientes a esa categoría.

        :data 
        Objeto devuelto, para cada variable, tenemos un diccionario de las categorias, y para categoria, los valores de cada muestra correspondientes a la variable y categoria.
        {
            variable1 : {
                categoria1: [muestra1[variable1], muestra2[variable1], muestra3[variable1]...], 
                categoria2: [muestra5[variable1], muestra6[variable1], muestra7[variable1]....],
            }
        }
        """
        data = collections.defaultdict(dict)
        columns = self.columns
        target = self.category_column_index
        dataset_category = {}
        for category in self.categories:
            dataset_category[category] = normalized_data[normalized_data[target] == category]
            for var in columns:
                data[var][category] = dataset_category[category][var]
        return data

    def get_kde(self, categories_by_var):
        """
        Calcula la densidad de probabilidad estimada para cada variable en cada categoria del dataset.

        :param categories_by_var
        Diccionario que tiene como clave una variable del dataset y como valor otro diccionario con las categorias como clave y las muestras pertenecientes a esa clase como valor.
        {
            variable1 : {
                categoria1: [muestra1[variable1], muestra2[variable1], muestra3[variable1]...], 
                categoria2: [muestra5[variable1], muestra6[variable1], muestra7[variable1]....],
            }
        }
        :param kde_bins:
        Número de bins que se usarán para el KDE.

        :kde_results  
        Diccionario que tiene como clave cada variable del dataset y como valor otro diccionario que tiene como clave cada una de las posibles categorias del dataset y como valor otro diccionario
        que tiene como clave 'x' y 'y' que corresponden al eje x y al eje y de la densidad estimada respectivamente.

        {
            variable1 : {
                categoria1: {x: KDE_var1_cat1_x, y: KDE_var1_cat1_y}, 
                categoria2: {x: KDE_var1_cat2_x, y: KDE_var1_cat2_y},
            }
        }
        """

        kde_results  = {}
        for var in categories_by_var:
            var_kde = collections.defaultdict(dict)
            for category in categories_by_var[var]:
                data = categories_by_var[var][category]
                kde = stats.gaussian_kde(data, bw_method = 'silverman' )
                ind = np.linspace(0,1,self.kde_bins)
                kde_pdf = kde.evaluate(ind)
                var_kde[category] = {'x': ind, 'y': kde_pdf}
            kde_results [var] = var_kde
        return kde_results 


    """ Este metodo calcula para cada gen, el vector de nuestra clasificacion, cada gen, compara los kde de cada categoria, dando para cada valor
    x del kde, la categoria de mayor valor"""
    def transform_kde_own_representation(self, kde):
        """
        Calcula para cada variable un vector de representacion, basado en los kde de cada categoria de la variable.
        Este nuevo vector de representacion, se calcula comparando los valores de los kde generados para cada categoria.
        Para cada variable comparamos los kde de todas sus categorias. Para cada posicion de los kde, obtenemos un nuevo valor, el cual es representado como las categorias ordenadas
        por valor del kde en esa posicion.

        Cada punto del vector de representacion, es una lista ordenada de las categorias de la variable, ordenadas segun el valor del kde en cada posicion para cada categoria, 
        obtenidos por el metodo get_var_kde_categories

        :param kde
        Diccionario generado por el metodo get_kde. Para cada variable el kde de cada categoria.
        {
            variable1 : {
                categoria1: KDE_var1_cat1, 
                categoria2: KDE_var1_cat2,
            }
        }

        :type_dictionary 
        Para cada variable, tenemos el nuevo vector de representacion, el cual para cada punto del KDE es la concatenacion de los nombres de las categorias, de mayor a menor valor.

        {
            variable1 : [[cat1,cat2], [cat1,cat2], [cat1,cat2], [cat2,cat1], [cat2,cat1], [cat1,cat2],.... ],
        }

        """
        var_data = kde[list(kde.keys())[0]]
        category_data = list(var_data.values())[0]
        data_len = len(category_data['y'])

        type_dictionary = {}
        for var in kde:
            type_dictionary[var] = [
                self.get_var_kde_categories(kde[var], index)
                for index in range(data_len)
            ]

        return type_dictionary

    def normalize_dataset(self, dataset):
        """
        Normaliza el dataset recibido entre 0 y 1.
        
        :param dataset:
            Dataset de pandas de los datos de las variables del csv de datos.
        """
        # Elimina la columna de categorias del dataset
        cat_value = dataset.pop(self.category_column_index)
        # Normaliza el dataset
        normalized_df = (dataset - dataset.min()) / ( dataset.max() - dataset.min())
        # Incluye la columna de categorias
        normalized_df[self.category_column_index] = cat_value
        return normalized_df

    def normalize_distance_pairs(self):
        """
        Normaliza los pares de los datos en self.pairs_by_distance entre 0 y 1.
        """
        keys = list(self.pairs_by_distance.keys())
        keys.sort()
        df_max = max(keys)
        normalized = [key / df_max for key in keys]
        self.pairs_by_distance = {element1: self.pairs_by_distance[element2] for element1, element2 in zip(normalized, keys)}


# TODO esto me encanta como esta aqui explicado
    def get_var_kde_categories(self, var_kde, index):
        """
        Obtiene el vector de representación para una posición determinada del KDE de una variable. El vector de representación
        se compone de las categorías ordenadas por el valor del KDE en la posición dada.

        Por ejemplo, si se tiene la variable x y dos categorías (cat1 y cat2), y en la posición 0 el valor del KDE de la 
        cat1 es mayor que el de la cat2, el vector devuelto para ese punto sería [cat1, cat2].

        Args:
            var_kde (dict): Un diccionario que contiene el KDE de cada categoría de una variable.
            index (int): La posición del KDE para la que se va a calcular el nuevo valor del KDE.

        Returns:
            list: Una lista ordenada de categorías según su valor de KDE en la posición dada.
        """
        # Crear un diccionario de categorías y valores de KDE para la posición dada
        kde_dict = {category: var_kde[category]['y'][index] for category in var_kde}
        # Ordenar las categorías según su valor de KDE en la posición dada
        sorted_categories = sorted(kde_dict.items(), key=lambda x: x[1])
        # Devolver una lista ordenada de las categorías
        return [x[0] for x in sorted_categories]


    def get_distance_pair_vars(self,var1_data, var2_data, var1, var2, max_distance_save):
        """
        Calcula la distancia entre dos variables utilizando su vector de representación basado en el KDE.

        La distancia se calcula sumando 1 por cada categoría que difiere en el orden entre las variables.
        Por ejemplo, si para una posición del vector de representación, la variable 1 tiene la categoría A en primer lugar y la variable 2 tiene la categoría B en primer lugar, se suma 1 a la distancia.
        Si la distancia total es mayor a cero, se normaliza dividiendo entre la longitud del vector de representación.

        Para evitar calcular la misma distancia más de una vez, se almacenan las distancias previamente calculadas en self.distance_memory.
        Además, se almacena información sobre los pares de variables que tienen la misma distancia en self.pairs_by_distance.

        Args:
            var1 (str): El nombre de la primera variable.
            var2 (str): El nombre de la segunda variable.

        Returns:
            float: La distancia entre las dos variables.
        """
        if var1 == var2:
            # Si se intenta comparar una variable consigo misma, la distancia es cero.
            return 0
       
        # Calcula la distancia sumando 1 por cada categoría que difiere en el orden entre las variables.
        distance = 0
        for vector_point1, vector_point2 in zip(var1_data, var2_data):
            length = len(vector_point1)
            for point1, point2 in zip(vector_point1, vector_point2):
                if point1 != point2:
                    distance += length
                    break
                length -= 1
        
        if(distance <= max_distance_save):
            self.pairs_by_distance.setdefault(distance, []).append([var1, var2])    
        else:
            self.pairs_by_distance.setdefault(distance, [])
    
        # Devuelve la distancia calculada.
        return distance 

    def get_umbral(self, umbral_index):
        distances = sorted(self.pairs_by_distance.keys())
        return distances[umbral_index]


    def get_complete_graph(self, umbral_index):
        """
        Este método dibuja el grafo de relaciones para las variables, donde cada nodo es una variable (en este caso, un gen) y las aristas representan la distancia entre ellos.
        Solo se dibujan aristas para las relaciones con una distancia menor al umbral.

        En este método se utiliza la variable self.pairs_by_distance, la cual almacena las relaciones existentes entre nodos para cada distancia posible. Al tener como keys
        las distancias posibles y como valor los pares de variables que pertenecen a esa distancia, podemos iterar por las posibles distancias ordenadas y obtener directamente
        los pares relacionados. Una vez que la distancia es mayor al umbral, se termina la iteración, ya que el resto de relaciones siempre tendrán una distancia mayor, lo que
        evita iterar por los pares de variables para obtener la distancia entre ellos.

        :return: Un diccionario que contiene para cada nodo su camino de menor peso hacia sus vecinos.
        """
        # Obtenemos el umbral de distancia a partir del método get_umbral
        self.draw_umbral = self.get_umbral(umbral_index)

        # Diccionario que almacena para cada nodo su camino de menor peso hacia sus vecinos.
        g1_edges_from_node = {}

        # Creamos el objeto Network para representar el grafo
        got_net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white')
        got_net.barnes_hut()

        # Iteramos por las posibles distancias ordenadas
        for distance in self.pairs_by_distance:
            # Si la distancia es mayor al umbral, terminamos la iteración
            if distance > self.draw_umbral:
                break

            # Iteramos por los pares de variables que pertenecen a esa distancia
            for src, dst in self.pairs_by_distance[distance]:
                # Añadimos los nodos si no existen aún en el grafo
                try:
                    got_net.get_node(src)
                except:
                    got_net.add_node(src, src, title=str(src))
                try:
                    got_net.get_node(dst)
                except:
                    got_net.add_node(dst, dst, title=str(dst))

                # Añadimos la arista con su peso
                weight = distance
                got_net.add_edge(src, dst, value=weight, weight=weight)
                # Actualizamos el diccionario g1_edges_from_node con el camino de menor peso para cada nodo
                if src not in g1_edges_from_node or g1_edges_from_node[src]['weight'] > weight:
                    g1_edges_from_node[src] = {"node": dst, "weight": weight}
                if dst not in g1_edges_from_node or g1_edges_from_node[dst]['weight'] > weight:
                    g1_edges_from_node[dst] = {"node": src, "weight": weight}

        neighbor_map = got_net.get_adj_list()
        for node in got_net.nodes:
            node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
            node['value'] = len(neighbor_map[node['id']])
        got_net.show_buttons(filter_=['physics'])
        graph_name = f'./outputs/{dataset_name}/graphs/graph_complete_{dataset_name}_umbral{round(self.draw_umbral, 2)}_nodes{self.max_nodos_network}_group{self.min_nodos_per_group}.html'
        if os.path.exists(graph_name):
            os.remove(graph_name)
        got_net.save_graph(graph_name)
        return g1_edges_from_node

    def get_validation_document(self, edges_from_nodem, graph_name):
        cont = 0
        for node in edges_from_nodem:
            f = open(f'./outputs/validation/validation_{graph_name}.txt', "w+")
            txt = re.sub('[{\'}\[\]]', '', str(list(map(lambda x: x[0], edges_from_nodem[node]))))
            f.write(f'{node}, {txt}')
            f.close()
            cont = cont + 1

    def get_minimun_graph(self, nodes_edges):
        """
        Obtiene el grafo de recubrimiento mínimo a partir de las relaciones mínimas obtenidas en el grafo completo.

        :param nodes_edges: dict
            Relaciones de nodos con su nodo más cercano y el peso de la arista que los une
        :return: pyvis.network.Network
            El grafo de recubrimiento mínimo
        """
        minimum_net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white')
        minimum_net.barnes_hut()

        # Para evitar relaciones circulares, recorremos los nodos y comprobamos si tienen algún nodo vecino.
        # Si no tienen ninguno, les añadimos una relación con el nodo de menor distancia.
        # Si ya tienen uno, no hacemos nada ya que tienen una arista de menor distancia.
    
        for node in nodes_edges.keys():
            try:
                neighbor = minimum_net.neighbors(node)
            except:
                neighbor = None
            if not neighbor or not node in neighbor:
                minimum_net.add_node(node, title=str(node), font={"size": 20})
                minimum_net.add_node(nodes_edges[node]['node'], title=str(nodes_edges[node]['node']), font={"size": 2000})
                minimum_net.add_edge(node, nodes_edges[node]['node'], value=nodes_edges[node]['weight'], weight=nodes_edges[node]['weight'])

        # minimum_net.show_buttons(filter_=['physics'])
        # minimum_net.show(f'graph_min_{self.draw_umbral}.html')

        return self.remove_small_connected_components(minimum_net)


    def remove_node(self, net, node_id):
        node_index = None
        for i, node in enumerate(net.nodes):
            if node['id'] == node_id:
                node_index = i
                break

        if node_index is not None:
            net.nodes.pop(node_index)
            net.edges = [edge for edge in net.edges if edge['from'] != node_id and edge['to'] != node_id]


    def remove_small_connected_components(self, net):
        visited = set()
        nodes_to_remove = []

        def dfs(node_id):
            visited.add(node_id)
            component_nodes = [node_id]
            neighbors = net.neighbors(node_id)
            for neighbor_id  in neighbors:
                if neighbor_id not in visited:
                    component_nodes.extend(dfs(neighbor_id))
            return component_nodes

        for node in net.nodes:
            node_id = node['id']
            if node_id not in visited:
                component_nodes = dfs(node_id)
                if len(component_nodes) < self.min_nodos_per_group:
                    nodes_to_remove.extend(component_nodes)

        for node_id in nodes_to_remove:
            self.remove_node(net, node_id)
        return net

    def get_network(self, umbral_index):
        """
        Retorna el grafo completo o el de recubrimiento mínimo según el valor del atributo 'complete_network'.
        
        Si 'complete_network' es True, se genera el grafo completo mediante la función 'get_complete_graph'
        y luego se obtiene el grafo de recubrimiento mínimo mediante 'get_minimum_graph'.
        
        Si 'complete_network' es False, se genera el grafo de recubrimiento mínimo mediante 'get_min_network'.
        
        Returns:
        - network: un objeto Network de la biblioteca vis.js que representa el grafo generado.
        """
        if self.complete_network:
            edges_from_node = self.get_complete_graph(umbral_index)
            network = self.get_minimun_graph(edges_from_node)
        else:
            g1_edges_from_node = self.get_min_network(umbral_index)
            network = self.get_minimun_graph(g1_edges_from_node)
        if(len(network.nodes) < self.max_nodos_network):
            return self.get_network(umbral_index+1)
        
        graph_name = f'./outputs/{dataset_name}/graphs/graph_{dataset_name}_umbral{round(self.draw_umbral, 2)}_nodes{self.max_nodos_network}_group{self.min_nodos_per_group}.html'
        if os.path.exists(graph_name):
            os.remove(graph_name)

        network.show_buttons(filter_=['physics'])
        network.save_graph(graph_name)
        return network
    
    def get_min_network(self, umbral_index):
        """
        Obtiene el grafo de recubrimiento mínimo sin generar el grafo de recubrimiento completo.
        El cálculo de los datos para el grafo mínimo es el mismo que para el completo, pero se omiten 
        los nodos que no cumplen con el umbral establecido.
        
        Retorna el grafo de recubrimiento mínimo.
        """
        # Calcula el umbral para el grafo mínimo.
        self.draw_umbral = self.get_umbral(umbral_index)
        distances = sorted(self.pairs_by_distance.keys())

        g1_edges_from_node = {}
        # Recorre las parejas ordenadas por distancia.
        for distance in distances:
            if distance > self.draw_umbral:
                break
            # Recorre las parejas de nodos que tienen la distancia actual.
            for src, dst in self.pairs_by_distance[distance]:
                weight = distance
                # Añade la arista si no existe o si su peso es menor.
                if src not in g1_edges_from_node or g1_edges_from_node[src]['weight'] > weight:
                    g1_edges_from_node[src] = {"node": dst, "weight": weight}
                if dst not in g1_edges_from_node or g1_edges_from_node[dst]['weight'] > weight:
                    g1_edges_from_node[dst] = {"node": src, "weight": weight}
        
        return g1_edges_from_node

    def export_validation_file(self, network):
        """
        Exporta el fichero para la validacion de los nodos. 

        :param nodes_edges
        Relaciones de los nodos y las aristas
        """
        nodos = network.nodes
        with open(f'./outputs/{dataset_name}/validation/todos_nodes{self.max_nodos_network}_group{self.min_nodos_per_group}.txt', 'w') as f:
            result = '\n'.join(str(nodo['id']) for nodo in nodos)
            f.write(result)

if __name__ == '__main__':
    project = Main()
    project.run()
