import os
import json
import glob
import networkx as nx
from pydantic import BaseModel, Field
from typing import Dict, List
import instructor
from anthropic import Anthropic
from KG_Extraction.micro import *
from onthology.onthology import *

# --- 1. SCHEMA PYDANTIC PER L'ENTITY RESOLUTION ---
class EntityMapping(BaseModel):
    mapping: Dict[str, str] = Field(
        ..., 
        description="Dizionario di normalizzazione. Chiave: ID originale estratto. Valore: ID canonico unificato. Es: {'char_il_portatore': 'char_frodo', 'char_frodo_baggins': 'char_frodo'}"
    )

# --- 2. FUNZIONE PER LA CHIAMATA AD HAIKU ---
import os
import instructor
from anthropic import Anthropic
from typing import List

# Presupponendo che EntityMapping sia già definita nel tuo file
# class EntityMapping(BaseModel): ...

def genera_mappa_entita_a_scaglioni(lista_nodi_grezzi: List[dict], batch_size: int = 250) -> dict:
    """
    Esegue l'Entity Resolution suddividendo i nodi in batch per non sforare 
    la context window dell'LLM. Mantiene una memoria degli ID canonici.
    """
    print(f"\nAvvio Entity Resolution scalabile. Nodi totali da analizzare: {len(lista_nodi_grezzi)}")
    client = instructor.from_anthropic(Anthropic(api_key=os.environ.get("API_KEY")))
    
    # Filtriamo subito per tenere solo Personaggi e Luoghi (risparmio token)
    nodi_filtrati = [n for n in lista_nodi_grezzi if n.get("node_type") in ["Character", "Location"]]
    
    mappa_globale = {}
    id_canonici_noti = set() # Questa è la "memoria" che passeremo di batch in batch
    
    # Dividiamo la lista in scaglioni (es. 250 nodi alla volta)
    for i in range(0, len(nodi_filtrati), batch_size):
        batch = nodi_filtrati[i:i + batch_size]
        numero_batch = (i // batch_size) + 1
        totale_batch = (len(nodi_filtrati) - 1) // batch_size + 1
        
        print(f"  -> Processando il Batch {numero_batch} di {totale_batch} (Nodi {i} a {i+len(batch)})...")
        
        # Trasformiamo il set di memoria in una stringa da inserire nel prompt
        memoria_str = ", ".join(id_canonici_noti) if id_canonici_noti else "Nessuna entità nota. Sei all'inizio del libro."
        
        SYSTEM_PROMPT_ER = f"""
        Sei un analista dati esperto in Entity Resolution per testi letterari.
        Stai processando un libro a scaglioni. 
        
        MEMORIA ENTITÀ GIÀ CONOSCIUTE (ID CANONICI):
        [{memoria_str}]
        
        Il tuo compito:
        Riceverai un nuovo batch di nodi estratti. Per ogni nodo:
        1. Se si riferisce a un'entità già presente nella MEMORIA qui sopra, mappalo a quell'esatto ID.
        2. Se si riferisce a un'entità NUOVA, inventa un NUOVO ID canonico logico e coerente.
        Restituisci la mappa completa per questo batch.
        """
        
        # Prepariamo l'input di questo specifico batch
        dati_input = "\n".join([f"ID: {n['id']} | Label: {n['label']} | Tipo: {n['node_type']}" for n in batch])
        
        # Chiamata API per il batch corrente
        try:
            risoluzione = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                response_model=EntityMapping,
                system=SYSTEM_PROMPT_ER,
                messages=[
                    {"role": "user", "content": f"Mappa queste entità:\n\n{dati_input}"}
                ],
                temperature=0.0
            )
            
            # 1. Aggiungiamo i risultati alla mappa globale
            mappa_globale.update(risoluzione.mapping)
            
            # 2. Aggiorniamo la memoria con i nuovi ID inventati per il prossimo ciclo
            for id_canonico in risoluzione.mapping.values():
                id_canonici_noti.add(id_canonico)
                
        except Exception as e:
            print(f"Errore durante il processamento del batch {numero_batch}: {e}")
            # In un ambiente di produzione reale potresti voler inserire un retry qui
            continue

    print(f"Risoluzione completata! Trovate {len(set(mappa_globale.values()))} entità uniche da {len(nodi_filtrati)} nodi grezzi.")
    return mappa_globale

# --- 3. FUNZIONE PRINCIPALE DI MERGING ---
def fondi_sub_grafi_dinamico(cartella_json: str):
    file_chunks = sorted(glob.glob(f"{cartella_json}/chunk_*.json"))
    
    # A. RACCOLTA PRELIMINARE DELLE ENTITÀ
    tutti_i_nodi_aperti = []
    for file_path in file_chunks:
        with open(file_path, 'r', encoding='utf-8') as f:
            dati_chunk = json.load(f)
            tutti_i_nodi_aperti.extend(dati_chunk.get("open_nodes", []))
            
    # B. GENERAZIONE DELLA MAPPA CON L'IA
    MAPPA_ENTITA_DINAMICA = genera_mappa_entita_a_scaglioni(tutti_i_nodi_aperti)
    print(f"Mappatura completata! Trovate {len(MAPPA_ENTITA_DINAMICA)} risoluzioni.")
    
    # Funzione helper interna per normalizzare un ID al volo
    def normalizza_id(nodo_id: str) -> str:
        return MAPPA_ENTITA_DINAMICA.get(nodo_id, nodo_id)
    
    # C. CREAZIONE DEL GRAFO GLOBALE
    G = nx.DiGraph()
    ultimo_evento_precedente = None
    
    for file_path in file_chunks:
        with open(file_path, 'r', encoding='utf-8') as f:
            dati_chunk = json.load(f)
            
        primo_evento_corrente = None
        ultimo_evento_corrente = None
        
        # 1. Inserimento Nodi Normalizzati
        for nodo in dati_chunk.get("open_nodes", []):
            id_canonico = normalizza_id(nodo["id"])
            G.add_node(id_canonico, label=nodo["label"], tipo=nodo["node_type"])
            
            if nodo["node_type"] == "Event":
                if primo_evento_corrente is None:
                    primo_evento_corrente = id_canonico
                ultimo_evento_corrente = id_canonico

        # 2. Inserimento degli Archi (Triplette)
        for rel in dati_chunk.get("relations", []):
            source_canonico = normalizza_id(rel.get("source_id"))
            
            # GESTIONE RETROCOMPATIBILITÀ: 
            # Controlliamo quale formato ha il JSON per estrarre il target
            if "target" in rel:
                # Caso A: Vecchi JSON di Moby Dick
                target_grezzo = rel["target"]
            elif rel.get("target_closed_value"):
                # Caso B: Nuovi JSON (Valore Chiuso da Enum)
                target_grezzo = rel["target_closed_value"]
            else:
                # Caso C: Nuovi JSON (Nodo Aperto)
                target_grezzo = rel.get("target_open_id")
                
            # Rete di sicurezza: se per qualsiasi motivo source o target sono vuoti, salta l'arco
            if not target_grezzo or not source_canonico:
                continue
                
            target_canonico = normalizza_id(target_grezzo)
            
            # Se il target è un "Valore Chiuso" o un concetto che non avevamo ancora nel grafo, creiamolo
            if not G.has_node(target_canonico):
                G.add_node(target_canonico, tipo="Valore_Chiuso")
                
            G.add_edge(source_canonico, target_canonico, label=rel.get("edge_label"))
            
        # 3. La Cucitura Temporale (Fabula)
        if ultimo_evento_precedente and primo_evento_corrente:
            G.add_edge(ultimo_evento_precedente, primo_evento_corrente, label="AVVIENE_DOPO")
            
        ultimo_evento_precedente = ultimo_evento_corrente
        
    print(f"Grafo Globale completato! Nodi totali: {G.number_of_nodes()}, Archi totali: {G.number_of_edges()}")
    return G
