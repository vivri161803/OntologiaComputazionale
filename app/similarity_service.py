import sys
import os
import torch
import torch.nn.functional as F

# Assicuriamoci che il modulo Inference e train_set siano raggiungibili
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Inference.book_embedder import BookGraphEmbedder
from train_set import train_params_3, train_library, test_library

class SimilarityEngine:
    """
    Classe OOP che espone un backend di comparazione semantica basato su GNN per i grafi di libri.
    """
    def __init__(self, model_weights="TrainedModel/narrative_kg_model_weights.pt", relation_mapping="TrainedModel/relation_mapping.json"):
        # Inizializza l'embedder (che carica il modello PyTorch una sola volta)
        # I path devono riferirsi alla directory root del progetto OntologiaComputazionale
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        self.embedder = BookGraphEmbedder(
            model_weights_path=os.path.join(base_dir, model_weights),
            relation_mapping_path=os.path.join(base_dir, relation_mapping),
            params=train_params_3
        )
        self.library_tsvs = train_library + test_library
        self.pooling_idx = 3 # 'concat' pooling method index from original script

    def get_top_k(self, target_tsv: str, k: int = 3) -> list:
        """
        Calcola i K libri più vicini vettorialmente a quello fornito.
        """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        target_path = os.path.join(base_dir, target_tsv)
        
        try:
            target_embeddings = self.embedder.embed_book(target_path)
            target_vector = target_embeddings[self.pooling_idx].unsqueeze(0)
            target_vector = F.normalize(target_vector, p=2, dim=1)
        except Exception as e:
            print(f"[Errore GNN] Impossibile estrarre l'embedding per {target_tsv}: {e}")
            return []
            
        library_similarities = []
        
        # Computiamo gli score iterando la libreria
        for lib_file in self.library_tsvs:
            if lib_file == target_tsv:
                continue

            lib_path = os.path.join(base_dir, lib_file)
            try:
                emb = self.embedder.embed_book(lib_path)
                book_vector = emb[self.pooling_idx].unsqueeze(0)
                book_vector = F.normalize(book_vector, p=2, dim=1)
                
                sim = torch.mm(target_vector, book_vector.T).item()
                library_similarities.append({
                    "path": lib_file,
                    "score": round(sim * 100, 2) # Conversione in %
                })
            except Exception as e:
                pass # Salta silenziosamente i fallimenti per non bloccare l'app
                
        # Riordino logico per similarità decrescente e taglio ai Top-K
        library_similarities.sort(key=lambda x: x["score"], reverse=True)
        return library_similarities[:k]
