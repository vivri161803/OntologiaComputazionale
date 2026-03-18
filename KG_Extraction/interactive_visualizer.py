import pandas as pd
import networkx as nx
from pyvis.network import Network

class PyVisGraphVisualizer:
    """
    Modulo OOP per la generazione di mappe interattive in HTML a 
    partire da un dataset TSV di input.
    """
    def __init__(self, bgcolor: str = '#1a1a1a', font_color: str = 'white'):
        self.bgcolor = bgcolor
        self.font_color = font_color
        
    def _apply_styles(self, G: nx.DiGraph):
        for node in G.nodes():
            node_str = str(node)
            G.nodes[node]["label"] = node_str
            G.nodes[node]["title"] = f"Tipologia/ID: {node_str}"
            
            if node_str.startswith("char_"):
                G.nodes[node]["color"] = "#ff6666" 
                G.nodes[node]["shape"] = "dot"
            elif node_str.startswith("evt_") and "type" not in node_str:
                G.nodes[node]["color"] = "#66b3ff" 
                G.nodes[node]["shape"] = "square"
            elif node_str.startswith("loc_") and "type" not in node_str:
                G.nodes[node]["color"] = "#99ff99" 
                G.nodes[node]["shape"] = "triangle"
            elif node_str.startswith("obj_"):
                G.nodes[node]["color"] = "#ffcc99" 
                G.nodes[node]["shape"] = "star"
            elif "role_" in node_str or "type_" in node_str:
                G.nodes[node]["color"] = "#ffd700" 
                G.nodes[node]["shape"] = "diamond"
                G.nodes[node]["size"] = 25 
            else:
                G.nodes[node]["color"] = "#cccccc" 

    def visualize_from_tsv(self, tsv_path: str, output_html: str):
        print(f"Generazione del grafo interattivo per {tsv_path}...")
        df = pd.read_csv(tsv_path, sep="\t")
        df['label'] = df['relation']
        df['title'] = df['relation']
        
        G = nx.from_pandas_edgelist(
            df, source='head', target='tail', edge_attr=['label', 'title'], create_using=nx.DiGraph()
        )
        
        self._apply_styles(G)
        
        net = Network(height='800px', width='100%', bgcolor=self.bgcolor, font_color=self.font_color, 
                      directed=True, select_menu=True, filter_menu=True)
        net.force_atlas_2based(gravity=-50, spring_length=150, overlap=1)
        net.from_nx(G)
        net.show(output_html, notebook=False)
        print(f"Fatto! Apri il file '{output_html}' nel tuo browser.")

if __name__ == "__main__":
    visualizer = PyVisGraphVisualizer()
    visualizer.visualize_from_tsv("book/Animal_Farm/Animal_Farm.tsv", "AnimalFarm.html")
