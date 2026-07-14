import osmnx as osm
import networkx as nw
import os
import matplotlib.pyplot as plt

from map_annotations import annotate_pois, annotate_street_names

class GraphConstructor:
    def __init__(self, place_name):
        self.place_name = place_name
        self.graph = None

    def construct_graph(self):
        # Descarga el grafo de la ubicación especificada
        self.graph = osm.graph_from_place(self.place_name, network_type='drive', retain_all=False)
        return self.graph

    def construct_dirt_graph(self):
        # Descarga el grafo de la ubicación especificada sin filtrar por tipo de carretera
        self.graph = osm.graph_from_place(self.place_name, network_type='all', retain_all=False)
        return self.graph

    def construct_graph_cached(self, file_path, network_type='drive'):
        """
        Carga el grafo desde `file_path` si existe; si no, lo descarga de OSM
        y lo guarda ahí. Se guarda el grafo CRUDO (sin clean_graph), de modo
        que el archivo siempre equivale a una descarga fresca — y congela los
        datos de OSM: ediciones al mapa posteriores no alteran los resultados.
        """
        if os.path.exists(file_path):
            print(f"Grafo '{network_type}' cargado desde caché: {file_path}")
            return self.load_graph(file_path)

        print(f"Descargando grafo '{network_type}' desde OSM (primera vez)...")
        if network_type == 'drive':
            self.construct_graph()
        else:
            self.construct_dirt_graph()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.save_graph(file_path)
        print(f"Grafo guardado en {file_path}")
        return self.graph

    def save_graph(self, file_path):
        if self.graph is not None:
            osm.save_graphml(self.graph, file_path)
        else:
            raise ValueError("Graph has not been constructed yet.")

    def load_graph(self, file_path):
        if os.path.exists(file_path):
            self.graph = osm.load_graphml(file_path)
            return self.graph
        else:
            raise FileNotFoundError(f"No file found at {file_path}.")
        
    def graph_to_geodataframe(self):
        if self.graph is not None:
            gdf_nodes, gdf_edges = osm.graph_to_gdfs(self.graph)
            return gdf_nodes, gdf_edges
        else:
            raise ValueError("Graph has not been constructed yet.")
        
    def print_graph_info(self):
        if self.graph is not None:
            print(f"Graph for {self.place_name}:")
            print(f"Number of nodes: {self.graph.number_of_nodes()}")
            print(f"Number of edges: {self.graph.number_of_edges()}")
        else:
            raise ValueError("Graph has not been constructed yet.")
        
    def visualize_graph(self, pois=None, show_street_names=True):
        # pois: lista {lat, lon, category, name} (p. ej. ODMatrixBuilder.get_pois())
        if self.graph is not None:
            fig, ax = osm.plot_graph(self.graph, show=False, close=False)
            if show_street_names:
                # el bgcolor por defecto de osmnx es oscuro (#111111)
                annotate_street_names(ax, self.graph, dark=True)
            if pois:
                annotate_pois(ax, pois, dark=True)
            plt.show()
        else:
            raise ValueError("Graph has not been constructed yet.")
        
    def clean_graph(self):
        if self.graph is not None:
            self.graph.remove_edges_from(list(nw.selfloop_edges(self.graph, keys=True)))
            self.graph.remove_edges_from([(u, v, k) for u, v, k, d in self.graph.edges(keys=True, data=True) if d.get('length', 0) <= 0])
            self.graph.remove_edges_from([(u, v, k) for u, v, k, d in self.graph.edges(keys=True, data=True) if d.get('highway') == 'service'])
            self.graph.remove_nodes_from(list(nw.isolates(self.graph)))
        else:
            raise ValueError("Graph has not been constructed yet.")
        
        