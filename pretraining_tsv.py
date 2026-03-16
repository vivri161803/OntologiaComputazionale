import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

# Import locali
from GNN.BERT.bert_csv import NodeFeatureEncoder
from GNN.pretraining.NegativeSampling_csv import GraphNegativeSampler
from GNN.EncoderDecoder import NarrativeKGModel
from GNN.pretraining.train_loop import train_kg_model
from train_set import *

# Parametri modello
train_params = train_params_1
NUM_EPOCHS = 500

# --- A. Definisci i tuoi file TSV ---
# Sostituisci questo array con i percorsi dei tuoi TSV creati dall'LLM
tsv_files = train_library

# --- B. Estrai i vettori di testo (SBERT) ---
print("\n[1/6] Estrazione Feature Iniziali (SBERT)...")
encoder = NodeFeatureEncoder()
encoder.process_tsv(tsv_files) # ORA USIAMO process_tsv!
X_tensor = encoder.generate_feature_matrix() 

device = torch.device('cpu')
X_tensor = X_tensor.to(device)

# --- C. Genera il Dataset (Negative Sampling) ---
print(f"\n[2/6] Generazione Dataset con Negative Sampling (K={train_params["k_negatives"]})...")
sampler = GraphNegativeSampler()
sampler.load_tsv(tsv_files)
df_dataset_strings = sampler.generate_dataset(k_negatives=train_params["k_negatives"])

# --- D. Il "PONTE": Mappa stringhe -> interi per PyTorch ---
print("\n[3/6] Preparazione Tensori...")
node_to_idx = encoder.node_id_to_idx 
idx_to_node = encoder.idx_to_node_id

unique_relations = list(sampler.relations)
rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
idx_to_rel = {v: k for k, v in rel_to_idx.items()}

df_dataset = df_dataset_strings.copy()
df_dataset['head'] = df_dataset['head'].map(node_to_idx)
df_dataset['tail'] = df_dataset['tail'].map(node_to_idx)
df_dataset['relation'] = df_dataset['relation'].map(rel_to_idx)
df_dataset = df_dataset.dropna().astype({'head': 'int64', 'tail': 'int64', 'relation': 'int64', 'label': 'int64'})

true_triplets_df = df_dataset[df_dataset['label'] == 1]

# --- E. Inizializza e Addestra il Modello ---
print("\n[4/6] Inizializzazione Modello Ottimizzato...")
model = NarrativeKGModel(
    num_nodes=len(node_to_idx), 
    num_relations=len(rel_to_idx), 
    in_channels=X_tensor.shape[1], 
    hidden_channels=train_params["hidden_channels"],
    num_layers=train_params["num_layers"],
    dropout_rate=train_params["dropout_rate"]
).to(device=device)

print("\n[5/6] Avvio Addestramento Definitivo...")
trained_model, training_history = train_kg_model(
    model=model, 
    df_dataset=df_dataset, 
    true_triplets_df=true_triplets_df, 
    node_features_X=X_tensor, 
    num_epochs=NUM_EPOCHS,
    lr=train_params["lr"],
    weight_decay=train_params["weight_decay"],
    margin=train_params["margin"]
)

# Salvataggio del Grafico Finale
plt.figure(figsize=(10, 6))
plt.plot(training_history['train_loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(training_history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('Andamento Loss Finale (Modello Ottimizzato)', fontsize=14)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('final_training_plot.png', dpi=300, bbox_inches='tight')

# --- F. ESTRAZIONE E SALVATAGGIO ---
print("\n[6/6] Estrazione e Salvataggio Embedding Finali...")
model.eval()
with torch.no_grad():
    edge_index = torch.tensor([true_triplets_df['head'].values, true_triplets_df['tail'].values], dtype=torch.long).to(device)
    edge_type = torch.tensor(true_triplets_df['relation'].values, dtype=torch.long).to(device)
    
    # final_node_embeddings contiene la fusione perfetta di Testo (SBERT) + Topologia (Ontologia)
    final_node_embeddings, final_rel_embeddings = model(X_tensor, edge_index, edge_type)

# 1. Salviamo i pesi del modello per inferenze future
os.makedirs("TrainedModel", exist_ok=True)
torch.save(model.state_dict(), 'TrainedModel/narrative_kg_model_weights.pt')

# 2. Salviamo i tensori degli embedding
torch.save(final_node_embeddings, 'TrainedModel/final_node_embeddings.pt')
torch.save(final_rel_embeddings, 'TrainedModel/final_rel_embeddings.pt')

# 3. Salviamo i dizionari di mappatura in formato JSON per poter ricollegare i vettori ai testi!
with open('TrainedModel/node_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(idx_to_node, f, indent=4)
    
with open('TrainedModel/relation_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(idx_to_rel, f, indent=4)

print("\n=== PIPELINE COMPLETATA CON SUCCESSO! ===")
print("Tutti i file sono stati salvati nella directory corrente.")
print(f"Dimensione Vettoriale dello Spazio Latente: {final_node_embeddings.shape[1]}")