import os
import json
import glob
import networkx as nx
import instructor
from anthropic import Anthropic
from typing import List, Dict
from onthology.onthology import EntityMapping

class GraphMerger:
    """
    Classe responsabile della fusione dei sub-grafi estratti e dell'Entity Resolution 
    per uniformare i nodi canonici tramite l'intelligenza artificiale.
    """
    def __init__(self, api_key: str, model_name: str = "claude-haiku-4-5-20251001", batch_size: int = 250):
        self.api_key = api_key
        self.client = instructor.from_anthropic(Anthropic(api_key=self.api_key))
        self.model_name = model_name
        self.batch_size = batch_size

    def _generate_entity_map_batch(self, raw_nodes: List[dict]) -> Dict[str, str]:
        """Elabora l'Entity Resolution in batch sequenziali per evitare limiti di contesto."""
        print(f"\nAvvio Entity Resolution scalabile. Nodi totali da analizzare: {len(raw_nodes)}")
        filtered_nodes = [n for n in raw_nodes if n.get("node_type") in ["Character", "Location"]]
        
        global_map = {}
        known_canonical_ids = set()
        
        for i in range(0, len(filtered_nodes), self.batch_size):
            batch = filtered_nodes[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(filtered_nodes) - 1) // self.batch_size + 1
            
            print(f"  -> Processando il Batch {batch_num} di {total_batches} (Nodi {i} a {i+len(batch)})...")
            memory_str = ", ".join(known_canonical_ids) if known_canonical_ids else "Nessuna entità nota. Sei all'inizio del libro."
            
            sys_prompt = (
                "Sei un analista dati esperto in Entity Resolution per testi letterari.\n"
                "Stai processando un libro a scaglioni.\n\n"
                "MEMORIA ENTITÀ GIÀ CONOSCIUTE (ID CANONICI):\n"
                f"[{memory_str}]\n\n"
                "Il tuo compito:\n"
                "Riceverai un nuovo batch di nodi estratti. Per ogni nodo:\n"
                "1. Se si riferisce a un'entità già presente nella MEMORIA qui sopra, mappalo a quell'esatto ID.\n"
                "2. Se si riferisce a un'entità NUOVA, inventa un NUOVO ID canonico logico e coerente.\n"
                "Restituisci la mappa completa per questo batch."
            )
            
            input_data = "\n".join([f"ID: {n['id']} | Label: {n['label']} | Tipo: {n['node_type']}" for n in batch])
            
            try:
                resolution = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    response_model=EntityMapping,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": f"Mappa queste entità:\n\n{input_data}"}],
                    temperature=0.0
                )
                global_map.update(resolution.mapping)
                known_canonical_ids.update(resolution.mapping.values())
            except Exception as e:
                print(f"Errore durante il processamento del batch {batch_num}: {e}")
                continue

        print(f"Risoluzione completata! Trovate {len(set(global_map.values()))} entità uniche.")
        return global_map

    def build_global_graph(self, json_dir: str) -> nx.DiGraph:
        """Legge la cartella dei chunk JSON, esegue l'ER e restituisce il DiGraph fuso e connesso."""
        file_chunks = sorted(glob.glob(f"{json_dir}/chunk_*.json"))
        
        all_open_nodes = []
        for file_path in file_chunks:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_open_nodes.extend(data.get("open_nodes", []))
                
        entity_map = self._generate_entity_map_batch(all_open_nodes)
        
        def _normalize_id(node_id: str) -> str:
            return entity_map.get(node_id, node_id)
            
        G = nx.DiGraph()
        last_previous_event = None
        
        for file_path in file_chunks:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            first_event_current = None
            last_event_current = None
            
            # Nodes Validation
            for node in data.get("open_nodes", []):
                canonical_id = _normalize_id(node["id"])
                G.add_node(canonical_id, label=node["label"], tipo=node["node_type"])
                
                if node["node_type"] == "Event":
                    if first_event_current is None:
                        first_event_current = canonical_id
                    last_event_current = canonical_id

            # Edges Generation
            for rel in data.get("relations", []):
                canon_source = _normalize_id(rel.get("source_id"))
                
                if "target" in rel:
                    raw_target = rel["target"]
                elif rel.get("target_closed_value"):
                    raw_target = rel["target_closed_value"]
                else:
                    raw_target = rel.get("target_open_id")
                    
                if not raw_target or not canon_source:
                    continue
                    
                canon_target = _normalize_id(raw_target)
                
                # Se il target non esiste nel grafo è un valore chiuso, creiamolo
                if not G.has_node(canon_target):
                    G.add_node(canon_target, tipo="Valore_Chiuso")
                    
                G.add_edge(canon_source, canon_target, label=rel.get("edge_label"))
                
            # Stitch Timeline (Fabula Cucitura)
            if last_previous_event and first_event_current:
                G.add_edge(last_previous_event, first_event_current, label="AVVIENE_DOPO")
                
            last_previous_event = last_event_current
            
        print(f"Grafo Globale completato! Nodi totali: {G.number_of_nodes()}, Archi totali: {G.number_of_edges()}")
        return G
