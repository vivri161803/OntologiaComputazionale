import json
import random
import pandas as pd
from typing import List
from KG_Extraction.books import *

class GraphNegativeSampler:
    def __init__(self, seed = 42):
        self.nodes_by_type = {
            "Character": [], "Event": [], "Location": [],
            "Role": [], "EventType": [], "LocationType": []
        }
        self.node_type_map = {}
        self.true_triplets = set()
        self.random_seed = seed
        
        # NUOVO: Set per collezionare dinamicamente tutti i ruoli relazionali
        self.partecipa_come_relations = set()
        
    def load_chunks(self, json_files: List[str]):
        """Carica i JSON e popola i dizionari e i set di relazioni."""
        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Estrazione Nodi Aperti
            for node in data.get("open_nodes", []):
                node_id, n_type = node["id"], node["node_type"]
                if node_id not in self.node_type_map:
                    self.nodes_by_type[n_type].append(node_id)
                    self.node_type_map[node_id] = n_type
                    
            # Estrazione Nodi Chiusi
            for node in data.get("closed_nodes", []):
                node_label, n_type = node["label"], node["node_type"]
                if node_label not in self.nodes_by_type[n_type]:
                    self.nodes_by_type[n_type].append(node_label)
                    
            # Estrazione Relazioni Vere e Vocabolario Archi
            for rel in data.get("relations", []):
                h = rel["source_id"]
                r = rel["edge_label"]
                t = rel["target_open_id"] if rel["target_open_id"] else rel["target_closed_value"]
                
                self.true_triplets.add((h, r, t))
                
                # NUOVO: Se è un arco di ruolo, lo salviamo nel nostro vocabolario
                if r.startswith("PARTECIPA_COME_"):
                    self.partecipa_come_relations.add(r)

    def _get_valid_tail_type(self, head_id: str, relation: str) -> str:
        """Restituisce il tipo di coda atteso (Ontologia)."""
        if relation == "AVVIENE_IN": return "Location"
        elif relation == "AVVIENE_DOPO": return "Event"
        elif relation == "PROVIENE_DA": return "Location"
        elif relation.startswith("PARTECIPA_COME_"): return "Event"
        elif relation == "COME":
            head_type = self.node_type_map[head_id]
            if head_type == "Character": return "Role"
            if head_type == "Event": return "EventType"
            if head_type == "Location": return "LocationType"
        raise ValueError(f"Relazione sconosciuta: {relation}")

    def generate_dataset(self, k_negatives: int = 3) -> pd.DataFrame:
        """Genera il dataset includendo la corruzione delle relazioni."""
        dataset = []
        
        for h, r, t in self.true_triplets:
            # Tripletta vera
            dataset.append({"head": h, "relation": r, "tail": t, "label": 1})
            
            for _ in range(k_negatives):
                random.seed(self.random_seed)
                # NUOVO: Decidiamo cosa corrompere. 
                # Usiamo pesi: 40% testa, 40% coda, 20% relazione.
                corrupt_target = random.choices(
                    ['head', 'tail', 'relation'], 
                    weights=[0.4, 0.4, 0.2]
                )[0]
                
                h_false, r_false, t_false = h, r, t
                valid_tail_type = self._get_valid_tail_type(h, r)
                head_type = self.node_type_map[h]
                
                if corrupt_target == 'head':
                    h_false = random.choice(self.nodes_by_type[head_type])
                    
                elif corrupt_target == 'tail':
                    t_false = random.choice(self.nodes_by_type[valid_tail_type])
                    
                elif corrupt_target == 'relation':
                    # Corrompiamo la relazione SOLO se è di tipo PARTECIPA_COME
                    # e ci assicuriamo di avere alternative disponibili
                    if r.startswith("PARTECIPA_COME_") and len(self.partecipa_come_relations) > 1:
                        # Rimuoviamo la relazione vera dalle opzioni e ne peschiamo una falsa
                        available_rels = list(self.partecipa_come_relations - {r})
                        r_false = random.choice(available_rels)
                    else:
                        # Fallback: se stiamo valutando un AVVIENE_IN o non abbiamo abbastanza
                        # ruoli estratti, corrompiamo la coda invece di fermarci.
                        t_false = random.choice(self.nodes_by_type[valid_tail_type])
                
                # Controllo anti-collisione
                if (h_false, r_false, t_false) not in self.true_triplets:
                    dataset.append({"head": h_false, "relation": r_false, "tail": t_false, "label": 0})
                    
        df = pd.DataFrame(dataset)
        return df.sample(frac=1).reset_index(drop=True)
    
# --- UTILIZZO ---
#sampler = GraphNegativeSampler()
#sampler.load_chunks(MobyDick) 
#df_training = sampler.generate_dataset(k_negatives=4)
#print(df_training.head(10))
#df_training.to_csv("MobyDick_ns.csv", index=False)