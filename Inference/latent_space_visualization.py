import torch
import pandas as pd
import json
import plotly.express as px
from sklearn.manifold import TSNE
import umap
from sentence_transformers import SentenceTransformer
from typing import Literal

from GNN.EncoderDecoder import NarrativeKGModel

class LatentSpaceVisualizer:
    """
    Riduce dimensionalmente lo spazio latente generato dalla Graph Neural Network
    e genera plot esplorabili interattivi 2D (T-SNE o UMAP) colorati per Categoria Logica.
    """
    def __init__(self, model_weights_path: str, relation_mapping_path: str, params: dict):
        self.device = torch.device('cpu')
        self.model_weights = model_weights_path
        self.params = params
        
        with open(relation_mapping_path, 'r', encoding='utf-8') as f:
            self.rel_to_idx = {v: int(k) for k, v in json.load(f).items()}
            
    def _assign_type(self, node_name: str) -> str:
        name = str(node_name)
        if name.startswith("char_"): return "Personaggio"
        elif name.startswith("loc_"): return "Luogo"
        elif name.startswith("evt_"): return "Evento"
        elif name.isupper(): return "Ruolo/Attributo"
        else: return "Altro"

    def _get_embeddings(self, tsv_path: str):
        df = pd.read_csv(tsv_path, sep='\t', names=['head', 'relation', 'tail'], dtype=str)
        df.dropna(subset=['head', 'relation', 'tail'], inplace=True)
        
        unique_nodes = sorted(list(set(df['head'].unique()).union(set(df['tail'].unique()))))
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        node_labels = [str(node).replace("_", " ") for node in unique_nodes]
        X_tensor = encoder.encode(node_labels, convert_to_tensor=True).to(self.device)
        
        heads, tails, relations = [], [], []
        for _, row in df.iterrows():
            h, r, t = str(row['head']), str(row['relation']), str(row['tail'])
            if h.lower() == 'nan' or r.lower() == 'nan' or t.lower() == 'nan': continue
            if r in self.rel_to_idx:
                heads.append(node_to_idx[h])
                relations.append(self.rel_to_idx[r])
                tails.append(node_to_idx[t])
                
        edge_index = torch.tensor([heads, tails], dtype=torch.long).to(self.device)
        edge_type = torch.tensor(relations, dtype=torch.long).to(self.device)
        
        model = NarrativeKGModel(
            num_nodes=len(unique_nodes), 
            num_relations=len(self.rel_to_idx), 
            in_channels=384, 
            hidden_channels=self.params["hidden_channels"],
            num_layers=self.params["num_layers"],
            dropout_rate=0.0
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.model_weights, map_location=self.device))
        model.eval()
        
        with torch.no_grad():
            node_embeddings, _ = model(X_tensor, edge_index, edge_type)
        return node_embeddings.cpu().numpy(), unique_nodes

    def visualize(self, tsv_path: str, method: Literal['tsne', 'umap'] = 'tsne'):
        print(f"\n--- Generazione Mappa 2D {method.upper()} Interattiva per: {tsv_path} ---")
        embeddings_np, unique_nodes = self._get_embeddings(tsv_path)
        
        if method == 'tsne':
            perplexity = min(30, len(unique_nodes) - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        else:
            n_neighbors = min(15, len(unique_nodes) - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, n_components=2, random_state=42)
            
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        clean_names = [str(n).replace("char_", "").replace("loc_", "").replace("evt_", "").replace("_", " ").title() for n in unique_nodes]
        node_types = [self._assign_type(n) for n in unique_nodes]
        
        df_plot = pd.DataFrame({'X': embeddings_2d[:, 0], 'Y': embeddings_2d[:, 1], 'Nome Nodo': clean_names, 'Tipologia': node_types})
        book_name = tsv_path.split('/')[-1].replace('.tsv', '')
        
        fig = px.scatter(
            df_plot, x='X', y='Y', hover_name='Nome Nodo', color='Tipologia', custom_data=['Tipologia'], 
            title=f"Spazio Latente Semantico ({method.upper()}): {book_name}", template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold 
        )
        fig.update_traces(
            marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')),
            hovertemplate="<b>%{hovertext}</b><br>Categoria: %{customdata[0]}<extra></extra>"
        )
        
        output_name = f"{method}_colored_{book_name}.html"
        fig.write_html(output_name)
        print(f"Mappa generata! Apri il file '{output_name}' nel tuo browser.")

if __name__ == "__main__":
    from train_set import train_params_3
    visualizer = LatentSpaceVisualizer("TrainedModel/narrative_kg_model_weights.pt", "TrainedModel/relation_mapping.json", train_params_3)
    visualizer.visualize("book/TheGreatGatsby/TheGreatGatsby.tsv", method="tsne")
    visualizer.visualize("book/Animal_Farm/Animal_Farm.tsv", method="umap")
