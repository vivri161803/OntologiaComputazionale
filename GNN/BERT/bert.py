import json
import torch
from sentence_transformers import SentenceTransformer
from KG_Extraction.books import *

class NodeFeatureEncoder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Carichiamo il modello SBERT. 
        # 'all-MiniLM-L6-v2' produce vettori a 384 dimensioni. Veloce e performante.
        print(f"Caricamento modello {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        # Strutture dati per l'output
        self.node_id_to_idx = {}  # Mappa ID testuale -> Indice numerico per PyTorch
        self.idx_to_node_id = {}  # Mappa inversa
        self.node_labels = []     # Lista dei testi da passare al modello
        
    def process_chunks(self, json_files: list):
        """Estrae le label da tutti i nodi (aperti e chiusi) e crea la mappatura."""
        current_idx = 0
        
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Elaboriamo i Nodi Aperti (Personaggi, Eventi, Luoghi)
            for node in data.get("open_nodes", []):
                node_id = node["id"]
                label = node["label"]
                
                # Se è la prima volta che vediamo questo nodo, lo registriamo
                if node_id not in self.node_id_to_idx:
                    self.node_id_to_idx[node_id] = current_idx
                    self.idx_to_node_id[current_idx] = node_id
                    
                    # Usiamo la stringa descrittiva come testo da encodare
                    self.node_labels.append(label)
                    current_idx += 1
                    
            # Elaboriamo i Nodi Chiusi (Ruoli, Tipi di evento)
            for node in data.get("closed_nodes", []):
                # Per i nodi chiusi usiamo la label come ID (es. "SOGGETTO_EROE")
                node_id = node["label"] 
                label = node["label"].replace("_", " ").lower() # Puliamo il testo (es. "soggetto eroe")
                
                if node_id not in self.node_id_to_idx:
                    self.node_id_to_idx[node_id] = current_idx
                    self.idx_to_node_id[current_idx] = node_id
                    
                    self.node_labels.append(label)
                    current_idx += 1

    def generate_feature_matrix(self) -> torch.Tensor:
        """Passa tutti i testi nel LLM e restituisce il tensore X."""
        print(f"Generazione embedding per {len(self.node_labels)} nodi...")
        
        # encode() accetta una lista di stringhe e restituisce un array NumPy
        # convert_to_tensor=True lo trasforma direttamente in un tensore PyTorch
        x_tensor = self.encoder.encode(self.node_labels, convert_to_tensor=True)
        
        print(f"Matrice delle feature generata. Shape: {x_tensor.shape}")
        return x_tensor

# --- ESEMPIO DI UTILIZZO ---
encoder = NodeFeatureEncoder()
encoder.process_chunks(TheVerdict) 
X = encoder.generate_feature_matrix()

# Ora abbiamo la matrice X e il dizionario encoder.node_id_to_idx
# Ad esempio, per vedere il vettore di "Jack Gisburn":
idx = encoder.node_id_to_idx["char_jack_gisburn"]
vector = X[idx]
print(vector)