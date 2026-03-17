import torch
import pandas as pd
import json
import plotly.express as px
import umap  # <-- Nuova libreria per la riduzione dimensionale
from sentence_transformers import SentenceTransformer

# Importazioni locali 
from train_set import train_params_3
from GNN.EncoderDecoder import NarrativeKGModel

def assegna_tipologia(node_name: str) -> str:
    """Deduce la tipologia del nodo analizzandone il formato testuale/prefisso."""
    nome = str(node_name)
    if nome.startswith("char_"):
        return "Personaggio"
    elif nome.startswith("loc_"):
        return "Luogo"
    elif nome.startswith("evt_"):
        return "Evento"
    elif nome.isupper():
        return "Ruolo/Attributo"
    else:
        return "Altro"

def visualize_book_clusters_umap(tsv_path: str, model_weights_path: str, relation_mapping_path: str):
    print(f"\n--- Generazione Mappa 2D UMAP Interattiva per: {tsv_path} ---")
    device = torch.device('cpu')
    
    # 1. CARICAMENTO DATI E MODELLO
    with open(relation_mapping_path, 'r', encoding='utf-8') as f:
        rel_to_idx = {v: int(k) for k, v in json.load(f).items()}
        
    df = pd.read_csv(tsv_path, sep='\t', names=['head', 'relation', 'tail'], dtype=str)
    df.dropna(subset=['head', 'relation', 'tail'], inplace=True)
    
    unique_nodes = set(df['head'].unique()).union(set(df['tail'].unique()))
    unique_nodes = sorted(list(unique_nodes))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    node_labels = [str(node).replace("_", " ") for node in unique_nodes]
    X_tensor = encoder.encode(node_labels, convert_to_tensor=True).to(device)
    
    heads, tails, relations = [], [], []
    for _, row in df.iterrows():
        h, r, t = str(row['head']), str(row['relation']), str(row['tail'])
        if h.lower() == 'nan' or r.lower() == 'nan' or t.lower() == 'nan': continue
        if r in rel_to_idx:
            heads.append(node_to_idx[h])
            relations.append(rel_to_idx[r])
            tails.append(node_to_idx[t])
            
    edge_index = torch.tensor([heads, tails], dtype=torch.long).to(device)
    edge_type = torch.tensor(relations, dtype=torch.long).to(device)
    
    model = NarrativeKGModel(
        num_nodes=len(unique_nodes), 
        num_relations=len(rel_to_idx), 
        in_channels=384, 
        hidden_channels=train_params_3["hidden_channels"],
        num_layers=train_params_3["num_layers"],
        dropout_rate=0.0
    ).to(device)
    
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    
    # 2. ESTRAZIONE DEGLI EMBEDDING
    with torch.no_grad():
        node_embeddings, _ = model(X_tensor, edge_index, edge_type)
    embeddings_np = node_embeddings.cpu().numpy()
    
    # 3. RIDUZIONE DIMENSIONALE CON UMAP
    print(f"Calcolo UMAP su {len(unique_nodes)} nodi...")
    
    # Parametri chiave di UMAP:
    # - n_neighbors: (default 15) Valori bassi (es. 5) focalizzano sulla struttura locale (cluster stretti). 
    #                Valori alti (es. 50) fanno guardare all'algoritmo l'intero grafo globale.
    # - min_dist: (default 0.1) Controlla quanto i punti possono stare appiccicati tra loro. 
    #             Aumentalo (es. 0.3) se vedi punti troppo sovrapposti, abbassalo (es. 0.01) per cluster densi.
    
    n_neighbors_sicuro = min(15, len(unique_nodes) - 1) # Sicurezza se il libro ha pochissimi nodi
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors_sicuro,
        min_dist=0.3, # L'ho alzato leggermente per distanziare meglio i nomi nei pop-up
        n_components=2, 
        random_state=42 # Assicura che ri-eseguendo lo script la mappa sia identica
    )
    
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # 4. PREPARAZIONE DATI PER PLOTLY
    clean_names = [
        str(nome).replace("char_", "").replace("loc_", "").replace("evt_", "").replace("_", " ").title() 
        for nome in unique_nodes
    ]
    node_types = [assegna_tipologia(nome) for nome in unique_nodes]
    
    df_plot = pd.DataFrame({
        'X': embeddings_2d[:, 0],
        'Y': embeddings_2d[:, 1],
        'Nome Nodo': clean_names,
        'Tipologia': node_types 
    })
    
    nome_libro = tsv_path.split('/')[-1].replace('.tsv', '')
    
    # 5. CREAZIONE GRAFICO INTERATTIVO A COLORI
    fig = px.scatter(
        df_plot, 
        x='X', 
        y='Y', 
        hover_name='Nome Nodo',
        color='Tipologia', 
        custom_data=['Tipologia'], 
        title=f"Spazio Latente Semantico (UMAP): {nome_libro}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold 
    )
    
    fig.update_traces(
        marker=dict(size=14, opacity=0.85, line=dict(width=1, color='white')),
        hovertemplate="<b>%{hovertext}</b><br>Categoria: %{customdata[0]}<extra></extra>"
    )
    
    output_filename = f"umap_colored_{nome_libro}.html"
    fig.write_html(output_filename)
    print(f"Mappa UMAP colorata generata! Apri il file '{output_filename}' nel tuo browser.")

# =====================================================
# ESEMPIO DI UTILIZZO
# =====================================================
if __name__ == "__main__":
    pesi_modello = "TrainedModel/narrative_kg_model_weights.pt"
    mappatura_rel = "TrainedModel/relation_mapping.json"
    libro_da_esplorare = "book/Animal_Farm/Animal_Farm.tsv" 
    
    visualize_book_clusters_umap(libro_da_esplorare, pesi_modello, mappatura_rel)