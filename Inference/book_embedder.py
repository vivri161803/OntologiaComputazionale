import torch
import pandas as pd
import json
import re
from typing import Tuple, List
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from GNN.EncoderDecoder import NarrativeKGModel

class BookGraphEmbedder:
    """
    Gestisce l'inferenza della GNN su nuovi libri (TSV) per calcolare
    i Graph Embeddings (fingerprint semantica) tramite diverse strategie di Graph Pooling.
    """
    def __init__(self, model_weights_path: str, relation_mapping_path: str, params: dict, device: torch.device = torch.device('cpu')):
        self.device = device
        self.model_weights_path = model_weights_path
        self.params = params
        
        with open(relation_mapping_path, 'r', encoding='utf-8') as f:
            idx_to_rel = json.load(f)
            self.rel_to_idx = {v: int(k) for k, v in idx_to_rel.items()}
            
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_book(self, tsv_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        print(f"Elaborazione del nuovo libro: {tsv_path}")
        
        df = pd.read_csv(tsv_path, sep='\t', names=['head', 'relation', 'tail'], dtype=str)
        df.dropna(subset=['head', 'relation', 'tail'], inplace=True)
        
        unique_nodes = sorted(list(set(df['head'].unique()).union(set(df['tail'].unique()))))
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
        
        node_labels = [str(node).replace("_", " ") for node in unique_nodes]
        X_tensor = self.encoder.encode(node_labels, convert_to_tensor=True).to(self.device)
        
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
        
        model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        model.eval()
        
        with torch.no_grad():
            node_embeddings, _ = model(X_tensor, edge_index, edge_type)
            mean_pool = torch.mean(node_embeddings, dim=0)
            max_pool = torch.max(node_embeddings, dim=0)[0]
            sum_pool = torch.sum(node_embeddings, dim=0)

            degrees = torch.zeros(node_embeddings.size(0), device=self.device)
            h_idx, t_idx = edge_index[0], edge_index[1]
            degrees.scatter_add_(0, h_idx, torch.ones_like(h_idx, dtype=torch.float))
            degrees.scatter_add_(0, t_idx, torch.ones_like(t_idx, dtype=torch.float))
            degrees = degrees + 1e-6
            
            attention_weights = F.softmax(degrees, dim=0)
            weighted_embeddings = node_embeddings * attention_weights.unsqueeze(1)
            attention_pool = torch.sum(weighted_embeddings, dim=0)

            concat_pool = torch.cat([mean_pool, max_pool, sum_pool, attention_pool], dim=0)
            sum_fusion_pool = mean_pool + max_pool + sum_pool + attention_pool

        return mean_pool, max_pool, sum_pool, concat_pool, sum_fusion_pool, attention_pool

    @staticmethod
    def extract_names_from_paths(path_list: List[str]) -> List[str]:
        pattern = re.compile(r"/([^/]+)\.tsv$")
        names = []
        for path in path_list:
            match = pattern.search(path)
            if match:
                names.append(match.group(1))
        return names

    def generate_embeddings_dataset(self, book_paths: List[str], output_csv: str):
        books_names = self.extract_names_from_paths(book_paths)
        results = []
        
        for path, name in zip(book_paths, books_names):
            mean_p, max_p, sum_p, concat_p, sum_fusion_p, att_p = self.embed_book(path)
            results.append({
                "Titolo": name,
                "MeanPooling": mean_p.tolist(),
                "MaxPooling": max_p.tolist(),
                "SumPooling": sum_p.tolist(),
                "ConcatenationEmbeddings": concat_p.tolist(),
                "SumEmbeddings": sum_fusion_p.tolist(),
                "AttentionEmbedding": att_p.tolist()
            })
            
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"Dataset inferenza generato e salvato in {output_csv}")

if __name__ == "__main__":
    from train_set import train_library, train_params_3
    embedder = BookGraphEmbedder(
        model_weights_path="TrainedModel/narrative_kg_model_weights.pt", 
        relation_mapping_path="TrainedModel/relation_mapping.json",
        params=train_params_3
    )
    embedder.generate_embeddings_dataset(train_library, "Inference/Train.csv")
