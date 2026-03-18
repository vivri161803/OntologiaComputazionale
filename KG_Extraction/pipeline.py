import os
import time
from dotenv import load_dotenv

from KG_Extraction.extractor import TextChunkExtractor
from KG_Extraction.merger import GraphMerger
from KG_Extraction.exporter import GraphTSVExporter

class NarrativeKGPipeline:
    """
    Orchestratore principale che unisce Extractor, Merger ed Exporter
    in una singola pipeline end-to-end orientata agli oggetti.
    """
    def __init__(self, api_key: str):
        self.extractor = TextChunkExtractor(api_key=api_key)
        self.merger = GraphMerger(api_key=api_key)
        self.exporter = GraphTSVExporter()
        
    def run(self, input_txt_path: str, output_dir: str, output_tsv_path: str):
        print("=== INIZIO PIPELINE OOP DI ESTRAZIONE KNOWLEDGE GRAPH ===")
        
        if not os.path.exists(input_txt_path):
            print(f"\n[ERRORE] Il file '{input_txt_path}' non esiste.")
            return

        print(f"\n>>> FASE 1: Estrazione sub-grafi da {input_txt_path}...")
        start_time = time.time()
        self.extractor.process_full_book(input_txt_path, output_dir)
        print(f"Fase 1 completata in {round(time.time() - start_time, 2)} secondi.")

        print(f"\n>>> FASE 2: Risoluzione entità e cucitura della fabula...")
        start_time = time.time()
        global_graph = self.merger.build_global_graph(output_dir)
        print(f"Fase 2 completata in {round(time.time() - start_time, 2)} secondi.")

        print(f"\n>>> FASE 3: Esportazione del grafo in TSV...")
        self.exporter.export(global_graph, output_tsv_path)

        print(f"\n=== PIPELINE OOP COMPLETATA CON SUCCESSO! ===")

if __name__ == "__main__":
    load_dotenv()
    key = os.environ.get("API_KEY")
    pipeline = NarrativeKGPipeline(api_key=key)
    pipeline.run(
        input_txt_path="book/txt/TenderIsTheNight.txt", 
        output_dir="book/TenderIsTheNight", 
        output_tsv_path="book/TenderIsTheNight/TenderIsTheNight.tsv"
    )
