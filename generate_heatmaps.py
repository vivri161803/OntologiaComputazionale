import os
from Inference.book_embedder import BookGraphEmbedder
from Inference.similarity_analyzer import SemanticSimilarityAnalyzer
from train_set import train_library, test_library, train_params_3

def main():
    print("="*60)
    print(" Generazione Heatmap Similarità da Libreria Completa ")
    print("="*60 + "\n")
    
    pesi_modello = "TrainedModel/narrative_kg_model_weights.pt"
    mappatura_rel = "TrainedModel/relation_mapping.json"
    dataset_inferenza_csv = "Inference/csv/Total.csv"
    
    # 1. Combina i due set di libri presi dalle liste in train_set.py
    libreria_completa = train_library + test_library
    
    if not libreria_completa:
        print("[Avviso] Le liste train_library e test_library sono vuote!")
        return
        
    print(f"Libreria trovata: elaborazione di {len(libreria_completa)} romanzi.")
    
    # 2. Inizializza l'embedder per estrarre il vettore latente (GNN embeddings)
    embedder = BookGraphEmbedder(
        model_weights_path=pesi_modello,
        relation_mapping_path=mappatura_rel,
        params=train_params_3
    )
    
    # 3. Genera il dataset unificato con i vettori
    print("\n[FASE 1] Calcolo ed estrazione degli embeddings (Graph Pooling)...")
    os.makedirs("Inference/csv", exist_ok=True)
    try:
        embedder.generate_embeddings_dataset(libreria_completa, dataset_inferenza_csv)
    except Exception as e:
        print(f"\n[ERRORE] Generazione embeddings fallita: {e}")
        return
        
    # 4. Inizializza l'analyzer per calcolare la Cosine Similarity e renderizzare la Heatmap
    print("\n[FASE 2] Analisi Cosine Similarity e rendering visuale delle Heatmap...")
    os.makedirs("Inference/plot", exist_ok=True)
    try:
        analyzer = SemanticSimilarityAnalyzer(csv_path=dataset_inferenza_csv)
        analyzer.generate_all_plots(output_dir="Inference/plot")
        print("\n🎉 Operazione completata con successo!")
        print(f"I file grafici (.png) delle Heatmap sono disponibili nella cartella: 'Inference/plot/'")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n[ERRORE] Creazione heatmap fallita: {e}")

if __name__ == "__main__":
    main()
