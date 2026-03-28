import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

class DashboardVisualizer:
    """
    A class to create a comparative dashboard for multiple graphs.
    """
    def __init__(self, data_list):
        """
        data_list: list of tuples (graph_name, graph_obj, stats_dict)
        """
        self.data_list = data_list
        # Number of plots per given graph (rows)
        # 1. Graph Layout
        # 2. Degree Distribution
        # 3. Betweenness Hist
        # 4. PageRank Top N
        # 5. Adjacency Heatmap
        self.n_rows = 5
        self.n_cols = len(data_list)

    def plot_dashboard(self, save_path=None):
        if self.n_cols == 0:
            print("No data to visualize.")
            return

        fig, axes = plt.subplots(self.n_rows, self.n_cols, figsize=(6 * self.n_cols, 25))
        
        # Ensure axes is always a 2D array for consistent indexing
        if self.n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for col_idx, (name, G, stats) in enumerate(self.data_list):
            
            # Row 0: Graph Layout
            ax = axes[0, col_idx]
            self._plot_graph_layout(ax, G)
            ax.set_title(f"{name}", fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel("Graph Layout", fontsize=12)

            # Row 1: Degree Distribution
            ax = axes[1, col_idx]
            self._plot_degree_distribution(ax, stats['degrees'])

            # Row 2: Betweenness Centrality
            ax = axes[2, col_idx]
            self._plot_betweenness_distribution(ax, stats['betweenness'])

            # Row 3: PageRank Rank Plot (Top 20)
            ax = axes[3, col_idx]
            self._plot_pagerank_top(ax, stats['pagerank'])

            # Row 4: Adjacency Heatmap
            ax = axes[4, col_idx]
            self._plot_adjacency_heatmap(ax, stats['adjacency_matrix'])

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        else:
            plt.show()

    def _plot_graph_layout(self, ax, G):
        # We try spring layout. If graph is too big, this might take time.
        try:
            pos = nx.spring_layout(G, seed=42, k=0.15)
        except:
            pos = nx.random_layout(G)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    def _plot_degree_distribution(self, ax, degrees_dict):
        degrees = list(degrees_dict.values())
        sns.histplot(degrees, bins=30, kde=True, ax=ax, color='coral')
        ax.set_title("Degree Distribution")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")

    def _plot_betweenness_distribution(self, ax, betweenness_dict):
        bc_values = list(betweenness_dict.values())
        sns.histplot(bc_values, bins=30, kde=True, ax=ax, color='mediumseagreen')
        ax.set_title("Betweenness Centrality Dist.")
        ax.set_xlabel("Betweenness Centrality")
        ax.set_ylabel("Frequency")

    def _plot_pagerank_top(self, ax, pagerank_dict, top_n=15):
        # Sort nodes by PageRank
        sorted_pr = sorted(pagerank_dict.items(), key=lambda item: item[1], reverse=True)
        top_nodes = sorted_pr[:top_n]
        
        if not top_nodes:
            ax.text(0.5, 0.5, "No PageRank data", ha='center', va='center')
            return

        nodes, pr_values = zip(*top_nodes)
        
        # Clean labels if they are too long
        cleaned_nodes = [str(n)[:15] + ".." if len(str(n)) > 15 else str(n) for n in nodes]

        sns.barplot(x=list(pr_values), y=list(cleaned_nodes), hue=list(cleaned_nodes), legend=False, ax=ax, palette='viridis')
        ax.set_title(f"Top {top_n} Nodes by PageRank")
        ax.set_xlabel("PageRank Value")
        ax.set_ylabel("Node")

    def _plot_adjacency_heatmap(self, ax, adj_matrix):
        # Using a binary/continuous heatmap.
        # If the matrix is too large, we disable annotations and tick labels
        is_large = adj_matrix.shape[0] > 100
        sns.heatmap(adj_matrix, cmap='YlGnBu', ax=ax, 
                    cbar=True,
                    xticklabels=not is_large, 
                    yticklabels=not is_large)
        ax.set_title("Adjacency Matrix Heatmap")
        if is_large:
            ax.set_xticks([])
            ax.set_yticks([])
