import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

class NodeFeatureEncoder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Caricamento modello SBERT {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        self.node_id_to_idx = {}  
        self.idx_to_node_id = {}  
        self.node_labels = []     
        
    def process_tsv(self, tsv_files: list):
        """Legge le colonne head e tail dei TSV per trovare tutti i nodi unici."""
        unique_nodes = set()
        
        for file in tsv_files:
            # Assumiamo che il TSV non abbia un header testuale, ma solo i dati. 
            # Se ha un header (es. "Soggetto Relazione Oggetto"), aggiungi header=0.
            df = pd.read_csv(file, sep='\t', names=['head', 'relation', 'tail'], dtype=str)
            
            # Aggiungiamo tutti i soggetti e oggetti unici al set
            unique_nodes.update(df['head'].dropna().unique())
            unique_nodes.update(df['tail'].dropna().unique())
            
        # Assegniamo un indice a ogni nodo e prepariamo le stringhe per SBERT
        for idx, node in enumerate(sorted(list(unique_nodes))):
            self.node_id_to_idx[node] = idx
            self.idx_to_node_id[idx] = node
            
            # Pulizia per SBERT: "Capitano_Achab" -> "Capitano Achab"
            label = str(node).replace("_", " ")
            self.node_labels.append(label)

    def generate_feature_matrix(self) -> torch.Tensor:
        print(f"Generazione embedding per {len(self.node_labels)} nodi unici...")
        x_tensor = self.encoder.encode(self.node_labels, convert_to_tensor=True)
        print(f"Matrice delle feature generata. Shape: {x_tensor.shape}")
        return x_tensor