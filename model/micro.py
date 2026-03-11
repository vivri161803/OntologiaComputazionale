import os
import json
import instructor
from anthropic import Anthropic
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from onthology.onthology import * 

# 1. INIZIALIZZAZIONE DEL CLIENT CLAUDE
# "instructor.from_anthropic" avvolge il client Anthropic per forzare l'output strutturato
# client = instructor.from_anthropic(Anthropic(api_key=os.environ.get("API_KEY")))

# 2. IL SYSTEM PROMPT (Le regole d'ingaggio)
SYSTEM_PROMPT = """
Sei un esperto di narratologia computazionale. Il tuo compito è leggere un capitolo 
di un romanzo ed estrarre un Knowledge Graph rigoroso, compilando lo schema dati fornito.

REGOLE DI ESTRAZIONE:
- Estrai le Entità a Valore Aperto (Character, Event, Location) limitandoti ai fatti del testo.
- Usa esclusivamente gli Archi Direzionati predefiniti.
- Usa l'arco "COME" per collegare un'entità aperta a un Valore Chiuso (Role, EventType, LocationType).
- Ricostruisci la fabula del capitolo usando l'arco "AVVIENE_DOPO" tra gli Eventi.
- Ricorda: un personaggio ha un ruolo specifico solo all'interno di quell'evento esatto. 
  Il ruolo è fuso nel nome dell'arco (es. PARTECIPA_COME_SOGGETTO).
"""

# 3. FUNZIONE DI CHUNKING ALLO STATO DELL'ARTE (LangChain)
def suddividi_in_chunk(testo_libro: str) -> List[str]:
    # Il RecursiveCharacterTextSplitter divide il testo semanticamente
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=15000,   # Dimensione ideale per non diluire l'attenzione del modello
        chunk_overlap=1000, # Sovrapposizione per non troncare eventi a metà
        separators=["\n\n\n", "\n\n", "\n", " ", ""] # Ordine di priorità per il taglio
    )
    return splitter.split_text(testo_libro)

# 4. LA FUNZIONE DI ESTRAZIONE (Anthropic API)
def estrai_sub_grafo(testo_chunk: str, numero_chunk: int) -> SubGraphExtraction:
    print(f"Inizio estrazione del Chunk {numero_chunk}...")
    
    api_key = os.environ.get("API_KEY")
    client = instructor.from_anthropic(Anthropic(api_key=api_key))
    # L'API di Anthropic ha una sintassi leggermente diversa rispetto a OpenAI
    sub_grafo = client.messages.create(
        model="claude-haiku-4-5-20251001", # O il modello Claude che preferisci
        max_tokens=20000, # Token massimi in output
        response_model=SubGraphExtraction, # LA NOSTRA GABBIA PYDANTIC
        system=SYSTEM_PROMPT, # In Claude il system prompt va in un parametro dedicato
        messages=[
            {"role": "user", "content": f"Estrai il grafo da questo testo:\n\n{testo_chunk}"}
        ],
        temperature=0.1 # Temperatura bassa per massimizzare il determinismo
    )
    return sub_grafo

# 5. IL CICLO PRINCIPALE (L'Automazione)
def processa_libro_intero(percorso_file: str):
    # Leggiamo il testo
    api_key = os.environ.get("API_KEY")
    client = instructor.from_anthropic(Anthropic(api_key=api_key))
    with open(percorso_file, 'r', encoding='utf-8') as file:
        libro_completo = file.read()
    chunks = suddividi_in_chunk(libro_completo)
    # Creiamo una cartella per i risultati
    os.makedirs("sub_grafi_output", exist_ok=True)
    for i, testo_chunk in enumerate(chunks, start=1):
        # Chiamata all'LLM
        grafo_estratto = estrai_sub_grafo(testo_chunk, i)
        # Salvataggio del Sub-Grafo in JSON
        nome_file = f"sub_grafi_output/chunk_{i}.json"
        with open(nome_file, 'w', encoding='utf-8') as f:
            f.write(grafo_estratto.model_dump_json(indent=2)) 
        print(f"Salvataggio completato: {nome_file}")

# 6. NORMALIZZAZIONE ETICHETTE ENTITA APERTE
def genera_mappa_entita_con_haiku(lista_nodi_grezzi: List[dict]) -> dict:
    print("Avvio Entity Resolution con Claude Haiku...")
    client = instructor.from_anthropic(Anthropic(api_key=os.environ.get("API_KEY")))
    
    SYSTEM_PROMPT_ER = """
    Sei un analista dati esperto in Entity Resolution per testi letterari.
    Riceverai una lista di nodi (ID e Label) estratti da vari capitoli di un romanzo.
    Il tuo compito è capire quali ID si riferiscono alla stessa identica entità narrativa 
    (es. 'char_strider' e 'char_aragorn') e restituire una mappa che li converta tutti 
    in un unico ID canonico di tua invenzione (es. 'char_aragorn').
    Ignora i nodi di tipo 'Event', concentrati solo su 'Character' e 'Location'.
    """
    
    # Convertiamo la lista grezza in una stringa leggibile per il prompt
    dati_input = "\n".join([f"ID: {n['id']} | Label: {n['label']} | Tipo: {n['node_type']}" for n in lista_nodi_grezzi])
    
    # Chiamata API forzata con Pydantic
    risoluzione = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        response_model=EntityMapping, 
        system=SYSTEM_PROMPT_ER,
        messages=[
            {"role": "user", "content": f"Mappa queste entità e unificale:\n\n{dati_input}"}
        ],
        temperature=0.0 # Temperatura a zero per massima coerenza logica
    )
    return risoluzione.mapping