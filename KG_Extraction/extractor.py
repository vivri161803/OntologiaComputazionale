import os
import json
import instructor
from anthropic import Anthropic
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from onthology.onthology import SubGraphExtraction

class TextChunkExtractor:
    """
    Classe responsabile del chunking del testo e dell'estrazione semantica 
    dei sub-grafi tramite chiamate LLM (Anthropic Claude).
    """
    def __init__(self, api_key: str, model_name: str = "claude-haiku-4-5-20251001", chunk_size: int = 15000, chunk_overlap: int = 1000):
        self.api_key = api_key
        # instructor forza l'output formattato su base Pydantic
        self.client = instructor.from_anthropic(Anthropic(api_key=self.api_key))
        self.model_name = model_name
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", " ", ""]
        )
        
        self.system_prompt = (
            "Sei un esperto di narratologia computazionale. Il tuo compito è leggere un capitolo\n"
            "di un romanzo ed estrarre un Knowledge Graph rigoroso, compilando lo schema dati fornito.\n\n"
            "REGOLE DI ESTRAZIONE:\n"
            "- Estrai le Entità a Valore Aperto (Character, Event, Location) limitandoti ai fatti del testo.\n"
            "- Usa esclusivamente gli Archi Direzionati predefiniti.\n"
            "- Usa l'arco 'COME' per collegare un'entità aperta a un Valore Chiuso (Role, EventType, LocationType).\n"
            "- Ricostruisci la fabula del capitolo usando l'arco 'AVVIENE_DOPO' tra gli Eventi.\n"
            "- Ricorda: un personaggio ha un ruolo specifico solo all'interno di quell'evento esatto.\n"
            "  Il ruolo è fuso nel nome dell'arco (es. PARTECIPA_COME_SOGGETTO)."
        )

    def split_text(self, full_text: str) -> List[str]:
        return self.text_splitter.split_text(full_text)
        
    def extract_sub_graph(self, text_chunk: str, chunk_index: int) -> SubGraphExtraction:
        """Invia un chunk all'LLM e ottiene in ritorno un modello Pydantic SubGraphExtraction."""
        print(f"Inizio estrazione del Chunk {chunk_index}...")
        return self.client.messages.create(
            model=self.model_name,
            max_tokens=20000,
            response_model=SubGraphExtraction,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"Estrai il grafo da questo testo:\n\n{text_chunk}"}
            ],
            temperature=0.1
        )
        
    def process_full_book(self, input_path: str, output_dir: str):
        """Legge l'intero libro, lo splitta in chunk e li processa sequenzialmente salvando i JSON."""
        with open(input_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
            
        chunks = self.split_text(full_text)
        os.makedirs(output_dir, exist_ok=True)
        
        for i, chunk_text in enumerate(chunks, start=1):
            sub_graph = self.extract_sub_graph(chunk_text, i)
            out_file = os.path.join(output_dir, f"chunk_{i}.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(sub_graph.model_dump_json(indent=2))
            print(f"Salvataggio completato: {out_file}")
