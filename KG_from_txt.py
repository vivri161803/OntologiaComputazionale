import os
import time
from KG_Extraction.micro import processa_libro_intero
from KG_Extraction.macro import fondi_sub_grafi_dinamico
from KG_Extraction.export import esporta_per_embedding
from dotenv import load_dotenv 
import anthropic

def main():
    load_dotenv()
    chiave = os.environ.get("API_KEY")
    client = anthropic.Anthropic(api_key=chiave)
    print("=== INIZIO PIPELINE DI ESTRAZIONE KNOWLEDGE GRAPH ===")
    
    # 1. Configurazione dei percorsi
    file_input = "book/txt/A_Vision_Of_Judgment.txt"
    output_dir = "book/A_Vision_Of_Judgment"
    file_output_tsv = "book/A_Vision_Of_Judgment.tsv"


    # Controllo di sicurezza: verifichiamo che il libro esista
    if not os.path.exists(file_input):
        print(f"\n[ERRORE] Il file '{file_input}' non è stato trovato nella cartella corrente.")
        print("Assicurati di aver scaricato il testo di Moby Dick e di averlo rinominato correttamente.")
        return

    # --- FASE 1: CHUNKING ED ESTRAZIONE LOCALE ---
    print(f"\n>>> FASE 1: Estrazione sub-grafi da {file_input}...")
    start_time = time.time()
    
    # Questa funzione dal tuo file micro.py si occupa di leggere, splittare e chiamare l'LLM
    # ATTENZIONE: Moby Dick è molto lungo. Questa operazione processerà l'intero libro.
    processa_libro_intero(file_input, output_dir) 
    
    print(f"Fase 1 completata in {round(time.time() - start_time, 2)} secondi.")

    # --- FASE 2: MERGING GLOBALE ED ENTITY RESOLUTION ---
    print(f"\n>>> FASE 2: Risoluzione entità e cucitura della fabula...")
    start_time = time.time()
    
    # Questa funzione dal tuo file macro.py legge i JSON, unifica i nodi e crea il DiGraph
    grafo_globale = fondi_sub_grafi_dinamico(output_dir)
    
    print(f"Fase 2 completata in {round(time.time() - start_time, 2)} secondi.")

    # --- FASE 3: ESPORTAZIONE PER EMBEDDING ---
    print(f"\n>>> FASE 3: Esportazione del grafo in TSV...")
    
    # Questa funzione dal tuo file export.py schiaccia il testo ed esporta le triplette
    esporta_per_embedding(grafo_globale, file_output_tsv)

    print(f"\n=== PIPELINE COMPLETATA CON SUCCESSO! ===")
    print(f"Il file pronto per l'addestramento KGE è: {file_output_tsv}")

if __name__ == "__main__":
    main()