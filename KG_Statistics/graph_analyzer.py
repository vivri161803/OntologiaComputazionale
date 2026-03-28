import csv
import networkx as nx

class GraphAnalyzer:
    """
    A class to analyze a knowledge graph from a TSV file.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.graph = nx.Graph()  # Using undirected graph for general stats, or DiGraph if directed. 
        # TSV usually has head, relation, tail. A generic knowledge graph is directed.
        self.is_directed = True 

    def load_graph(self):
        """
        Loads the graph from the TSV file. 
        Assumes columns: head, relation, tail.
        """
        if self.is_directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None) # Skip header if present (e.g., 'head\trelation\ttail')
            for row in reader:
                if len(row) >= 3:
                    head, relation, tail = row[0], row[1], row[2]
                    self.graph.add_edge(head, tail, relation=relation)
        return self.graph

    def compute_statistics(self):
        """
        Computes various statistics for the graph:
        - node degrees
        - betweenness centrality
        - pagerank
        - adjacency matrix
        """
        if len(self.graph.nodes) == 0:
            raise ValueError("The graph is empty. Please load_graph() first.")

        # 1. Degrees
        if self.is_directed:
            degrees = dict(self.graph.in_degree()) # Or use out_degree or degree
            # Let's use total degree for the histogram
            total_degrees = dict(self.graph.degree())
        else:
            total_degrees = dict(self.graph.degree())

        # 2. Betweenness Centrality
        # This can be slow for very large graphs
        betweenness = nx.betweenness_centrality(self.graph)

        # 3. PageRank
        pagerank = nx.pagerank(self.graph)

        # 4. Adjacency Matrix
        # returns a scipy sparse matrix, we convert it to dense format for heatmap
        nodelist = list(self.graph.nodes)
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=nodelist)

        stats = {
            'degrees': total_degrees,
            'betweenness': betweenness,
            'pagerank': pagerank,
            'adjacency_matrix': adj_matrix,
            'nodelist': nodelist
        }
        return stats
