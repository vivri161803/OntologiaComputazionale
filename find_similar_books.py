import torch
import torch.nn.functional as F
from typing import List

from Inference.book_embedder import BookGraphEmbedder
from train_set import train_library, test_library, train_params_3

def find_top_k_similar_books(target_tsv: str, library_tsvs: List[str], k: int = 3, pooling_method: str = 'concat'):
    """
    Data la path in input di un libro (TSV), estrapola il suo embedding tramite la GNN 
    e lo confronta con tutti gli altri libri della libreria per trovare i Top-K più simili.
    """
    print(f"\nCerco i {k} romanzi più narratologicamente affini a: {target_tsv}\n")
    
    # Inizializziamo l'embedder usando la classe sviluppata nel refactoring
    embedder = BookGraphEmbedder(
        model_weights_path="TrainedModel/narrative_kg_model_weights.pt", 
        relation_mapping_path="TrainedModel/relation_mapping.json",
        params=train_params_3
    )
    
    try:
        # 1. Calcoliamo l'embedding del libro target
        print("Calcolo embedding del libro in input...")
        target_embeddings = embedder.embed_book(target_tsv)
    except FileNotFoundError:
        print(f"\n[ERRORE] Il file {target_tsv} non è stato trovato!")
        return
        
    # Indici restituiti da embed_book: 0=mean, 1=max, 2=sum, 3=concat, 4=sum_fusion, 5=attention
    pooling_idx = {'mean': 0, 'max': 1, 'sum': 2, 'concat': 3, 'sum_fusion': 4, 'attention': 5}.get(pooling_method, 3)
    
    # Estraiamo il vettore corretto e sistemiamo la shape per la Cosine Similarity ([1, Dimensione])
    target_vector = target_embeddings[pooling_idx].unsqueeze(0)  
    target_vector = F.normalize(target_vector, p=2, dim=1)
    
    # 2. Calcoliamo l'embedding per tutta la libreria e li compariamo sul momento
    print("\nScansione della libreria (train + test set)...")
    library_similarities = []
    
    for book_tsv in library_tsvs:
        if book_tsv == target_tsv:
            continue # Evitiamo di confrontare il libro con sé stesso se fa già parte del test set!
            
        print(f"  Analizzo: {book_tsv}")
        try:
            emb = embedder.embed_book(book_tsv)
            book_vector = emb[pooling_idx].unsqueeze(0) 
            book_vector = F.normalize(book_vector, p=2, dim=1)
            
            # Calcolo Cosine Similarity (Prodotto scalare su vettori normalizzati)
            sim = torch.mm(target_vector, book_vector.T).item()
            
            nome_libro = embedder.extract_names_from_paths([book_tsv])[0]
            library_similarities.append((nome_libro, sim))
        except FileNotFoundError:
            print(f"  [Avviso] File {book_tsv} saltato (Non ancora estratto/generato).")
        except Exception as e:
            print(f"  [Errore] Impossibile processare {book_tsv}: {e}")
            
    # 3. Ordiniamo per similarità decrescente
    library_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Stampiamo i risultati in formato elegante
    target_name = embedder.extract_names_from_paths([target_tsv])[0]
    print("\n" + "="*50)
    print(f"🏆 I {k} romanzi più simili a '{target_name}':")
    print("="*50)
    
    if not library_similarities:
        print("Nessun libro valido trovato nella libreria per eseguire la comparazione.")
        return
        
    for i in range(min(k, len(library_similarities))):
        nome, score = library_similarities[i]
        percent_score = score * 100
        print(f" {i+1}. {nome.ljust(30)} -> Affinità Strutturale: {percent_score:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Puoi cambiare questo percorso con qualsiasi file TSV estratto
    # Es. "book/TenderIsTheNight/TenderIsTheNight.tsv"
    libro_target = "book/Animal_Farm/Animal_Farm.tsv" 
    
    # Combiniamo entrambi i set dal nostro file di configurazione
    libreria_completa = train_library + test_library
    
    # Avvia l'algoritmo usando l'approccio di Pooling "concat"
    find_top_k_similar_books(target_tsv=libro_target, library_tsvs=libreria_completa, k=3, pooling_method='concat')
