import os
from dotenv import load_dotenv

# Importazione di tutte le classi OOP realizzate durante il refactoring
from KG_Extraction.pipeline import NarrativeKGPipeline
from KG_Extraction.interactive_visualizer import PyVisGraphVisualizer
from Inference.book_embedder import BookGraphEmbedder
from Inference.latent_space_visualization import LatentSpaceVisualizer
from Inference.similarity_analyzer import SemanticSimilarityAnalyzer

# Importazione dei parametri e library per la comparazione
from train_set import train_params_3, train_library, test_library

def main():
    # Caricamento chiavi API per 'Claude Haiku' dal file .env
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    
    if not api_key:
        print("[ERRORE] Assicurati di avere la variabile d'ambiente API_KEY configurata!")
        return
        
    print("\n" + "="*60)
    print("  🚀 DEMO END-TO-END: ONTOLOGIA COMPUTAZIONALE 🚀  ")
    print("="*60)

    # -------------------------------------------------------------
    # CONFIGURAZIONE DEI PERCORSI DEMO
    # -------------------------------------------------------------
    # Puoi cambiare 'TenderIsTheNight.txt' con un qualsiasi file TXT
    input_text_file = "book/txt/TenderIsTheNight.txt" 
    demo_book_dir = "book/DemoBook"
    output_tsv_file = f"{demo_book_dir}/DemoBook.tsv" 
    
    # Assicurati di avere allenato il modello e generato questi file in precedenza
    # (Altrimenti devi lanciare GNNTrainer prima di questo demo)
    pesi_modello = "TrainedModel/narrative_kg_model_weights.pt"
    mappatura_rel = "TrainedModel/relation_mapping.json"
    dataset_inferenza_csv = "Inference/csv/Demo_Dataset.csv"

    # -------------------------------------------------------------
    # FASE 1: Estrazione del Knowledge Graph dal file TXT -> TSV
    # -------------------------------------------------------------
    print("\n[STEP 1] Data Extraction: Traduzione Testo -> Grafo TSV")
    pipeline = NarrativeKGPipeline(api_key=api_key)
    
    # Esegue l'estrazione a chunk (Anthropic API), unifica logica ER e genera TSV
    pipeline.run(
        input_txt_path=input_text_file,
        output_dir=demo_book_dir,
        output_tsv_path=output_tsv_file
    )
    
    # (Se vuoi bypassare i costi API e la lunghezza d'estrazione per testare la demo al volo, 
    # commenta il pipeline.run qui sopra e usa un TSV finto o un testo più corto/in cache)

    # -------------------------------------------------------------
    # FASE 2: Visualizzazione del Grafo Interattivo Estratto
    # -------------------------------------------------------------
    print("\n[STEP 2] Visualizzazione Interattiva: Navigazione Nodi")
    visualizer_kg = PyVisGraphVisualizer()
    out_html_kg = "Demo_Narrative_Graph.html"
    visualizer_kg.visualize_from_tsv(output_tsv_file, out_html_kg)

    # -------------------------------------------------------------
    # FASE 3: Generazione Spazio Latente e Visualizzazione Nodi in 2D
    # -------------------------------------------------------------
    print("\n[STEP 3] Inferenza e Spazio Latente: UMAP & T-SNE")
    # Utilizziamo la GNN per generare lo specchio 2D (riduzione dimensione da 384->2)
    visualizer_latent = LatentSpaceVisualizer(
        model_weights_path=pesi_modello,
        relation_mapping_path=mappatura_rel,
        params=train_params_3
    )
    visualizer_latent.visualize(output_tsv_file, method="umap")
    visualizer_latent.visualize(output_tsv_file, method="tsne")

    # -------------------------------------------------------------
    # FASE 4: Calcolo Pooling Combinato e Similarità Inter-Testuale
    # -------------------------------------------------------------
    print("\n[STEP 4] Graph Embedding e Comparazione Similarità")
    embedder = BookGraphEmbedder(
        model_weights_path=pesi_modello,
        relation_mapping_path=mappatura_rel,
        params=train_params_3
    )
    
    # Aggiungiamo il nostro romanzo "Demo" al dataset di confronto (assieme a quelli di addestramento)
    book_list_to_compare = train_library +  test_library + [output_tsv_file] 
    
    # 4.1 - Calcoliamo i vettori Mean, Max, Sum, Attention per tutti e li uniamo in un CSV Pydantic ast-friendly
    os.makedirs("Inference/csv", exist_ok=True)
    embedder.generate_embeddings_dataset(book_list_to_compare, dataset_inferenza_csv)
    
    # 4.2 - Generiamo le Heatmap termiche calcolando matematicamente la Cosine Similarity
    os.makedirs("Inference/plot", exist_ok=True)
    analyzer = SemanticSimilarityAnalyzer(csv_path=dataset_inferenza_csv)
    analyzer.generate_all_plots(output_dir="Inference/plot")
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETATA CON SUCCESSO! 🎉")
    print(f"Controlla la root del progetto per aprire '{out_html_kg}' ed esplorare il grafo logico estratto.")
    print("Le mappe UMAP/T-SNE .html interattive sono pronte.")
    print("Visualizza le heatmap finali in 'Inference/plot/' per studiare le geometrie tra romanzi.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
