import torch
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import re

# importazioni locali 
from train_set import train_params_3
from train_set import test_library
from train_set import train_library
from GNN.EncoderDecoder import NarrativeKGModel

# 1. PARAMETRI del modello (Devono corrispondere esattamente al training)
HIDDEN_CHANNELS = train_params_3["hidden_channels"]
NUM_LAYERS = train_params_3["num_layers"]
DROPOUT_RATE = train_params_3["dropout_rate"] # Dropout a 0 durante l'inferenza!
IN_CHANNELS = 384  # Dimensione output di SBERT

def get_new_book_embedding(tsv_path: str, model_weights_path: str, relation_mapping_path: str):
    device = torch.device('cpu')
    
    print(f"Elaborazione del nuovo libro: {tsv_path}")
    
    # --- A. Carica il dizionario delle Relazioni ---
    with open(relation_mapping_path, 'r', encoding='utf-8') as f:
        # JSON salva le chiavi numeriche come stringhe, quindi le riconvertiamo in int
        idx_to_rel = json.load(f)
        rel_to_idx = {v: int(k) for k, v in idx_to_rel.items()}
    
    NUM_RELATIONS = len(rel_to_idx)
    
    # --- B. Processa il TSV e genera i vettori SBERT ---
    # Converti i NaN di pandas in stringhe vuote per evitare errori più avanti, 
    # poi li scarteremo esplicitamente
    df = pd.read_csv(tsv_path, sep='\t', names=['head', 'relation', 'tail'], dtype=str)
    
    # Pulisci il DataFrame scartando righe dove manca la testa, la coda o la relazione
    df.dropna(subset=['head', 'relation', 'tail'], inplace=True)
    
    # Trova i nodi unici e crea la mappatura
    unique_nodes = set(df['head'].unique()).union(set(df['tail'].unique()))
    unique_nodes = sorted(list(unique_nodes))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    
    print(f"Trovati {len(unique_nodes)} nodi unici. Calcolo SBERT...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # Puliamo i testi come facevamo nel training (es. Frodo_Baggins -> Frodo Baggins)
    node_labels = [str(node).replace("_", " ") for node in unique_nodes]
    X_tensor = encoder.encode(node_labels, convert_to_tensor=True).to(device)
    
    # --- C. Costruisci i tensori strutturali (edge_index, edge_type) ---
    heads, tails, relations = [], [], []
    relazioni_scartate = 0
    righe_nan_scartate = 0
    
    for _, row in df.iterrows():
        h, r, t = str(row['head']), str(row['relation']), str(row['tail'])
        
        # Filtro aggiuntivo: controlla le stringhe 'nan' letterali generate
        if h.lower() == 'nan' or r.lower() == 'nan' or t.lower() == 'nan':
            righe_nan_scartate += 1
            continue
            
        # FILTRO DI SICUREZZA: Teniamo solo le relazioni viste nel training
        if r in rel_to_idx:
            heads.append(node_to_idx[h])
            relations.append(rel_to_idx[r])
            tails.append(node_to_idx[t])
        else:
            relazioni_scartate += 1
            
    if relazioni_scartate > 0:
        print(f"Avviso: {relazioni_scartate} relazioni scartate perché non presenti nel training set.")
    if righe_nan_scartate > 0:
         print(f"Avviso: {righe_nan_scartate} righe scartate perché contenevano 'nan'.")
        
    edge_index = torch.tensor([heads, tails], dtype=torch.long).to(device)
    edge_type = torch.tensor(relations, dtype=torch.long).to(device)
    
    # --- D. Inizializza il Modello e carica i pesi ---
    # Nota: num_nodes serve solo per la firma della classe, 
    # ma RGCNConv lavora dinamicamente sulla shape di X_tensor.
    model = NarrativeKGModel(
        num_nodes=len(unique_nodes), 
        num_relations=NUM_RELATIONS, 
        in_channels=IN_CHANNELS, 
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Carichiamo i pesi addestrati
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval() # Fondamentale: spegne il dropout e fissa i pesi
    
    # --- E. Inferenza e Graph Pooling ---
    with torch.no_grad():
        # Otteniamo gli embedding dei nodi arricchiti dalla struttura narrativa
        node_embeddings, _ = model(X_tensor, edge_index, edge_type)
        
        # 1. MEAN POOLING (Media): Cattura il "Tono Generale" del libro
        mean_pool = torch.mean(node_embeddings, dim=0)
        
        # 2. MAX POOLING (Massimo): Cattura la presenza di archetipi forti/picchi
        max_pool = torch.max(node_embeddings, dim=0)[0]

        # 3. SUM POOLING 
        sum_pool = torch.sum(node_embeddings, dim=0)
        
        # Concateniamo i due vettori per un Graph Embedding ricchissimo
        graph_embedding_concat = torch.cat([mean_pool, max_pool, sum_pool], dim=0)
        # Sommiamo i due vettori per un Graph Embedding ricchissimo
        graph_embedding_sum = mean_pool + max_pool + sum_pool
        
    print(f"Graph Embedding generato con successo! Shape (Concat): {graph_embedding_concat.shape}")
    print(f"Graph Embedding generato con successo! Shape (Sum): {graph_embedding_sum.shape}")
    print("")
    return mean_pool, max_pool, sum_pool, graph_embedding_concat, graph_embedding_sum

# 2. Estrazione nomi a partire dal path 

def estrai_nomi_dal_path(path_list):
    pattern = re.compile(r"/([^/]+)\.tsv$")

    nomi_libri = []

    for path in path_list:
        match = pattern.search(path)
        if match:
            # group(1) prende esattamente quello che c'è tra le parentesi tonde della regex
            nome_libro = match.group(1)
            nomi_libri.append(nome_libro)

    return nomi_libri

# =====================================================
# ESEMPIO DI UTILIZZO
# =====================================================
if __name__ == "__main__":

    # parametri main
    output_file = "Inference/Test.csv"
    pesi_modello = "TrainedModel/narrative_kg_model_weights.pt"
    mappatura_rel = "TrainedModel/relation_mapping.json"
    
    nomi_libri = estrai_nomi_dal_path(train_library)
    
    risultati = []

    for path, nome in zip(test_library, nomi_libri):
        # Estraiamo l'impronta digitale dell'intero romanzo
        mean_pool, max_pool, sum_pool, vettore_libro_concat, vettore_libro_sum = get_new_book_embedding(path, pesi_modello, mappatura_rel)
        
        risultati.append({
             "Titolo" : nome,
             "MeanPooling" : mean_pool.tolist(),
             "MaxPooling" : max_pool.tolist(),
             "SumPooling" : sum_pool.tolist(),
             "ConcatenationEmbeddings" : vettore_libro_concat.tolist(),
             "SumEmbeddings" : vettore_libro_sum.tolist()
             
        })
        

    result = pd.DataFrame(risultati)

    result.to_csv(output_file, index=False)