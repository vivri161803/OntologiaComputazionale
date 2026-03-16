import optuna
import torch
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Import Locali ---
# Assicurati che i file bert.py e NegativeSampling.py siano aggiornati 
# con le funzioni process_tsv() e load_tsv() scritte precedentemente.
from GNN.BERT.bert_csv import NodeFeatureEncoder
from GNN.pretraining.NegativeSampling_csv import GraphNegativeSampler
from GNN.EncoderDecoder import NarrativeKGModel
from GNN.pretraining.train_loop import train_kg_model
from train_set import train_library

# --- A. Definisci i tuoi file TSV ---
# Inserisci i percorsi dei file TSV fattorizzati dall'LLM
tsv_files = train_library

# --- B. Estrai i vettori di testo (SBERT) una volta sola ---
print("Estrazione vettori semantici SBERT dai TSV...")
encoder = NodeFeatureEncoder()
encoder.process_tsv(tsv_files) # Legge i TSV e trova tutti i nodi unici
X_tensor = encoder.generate_feature_matrix() # Shape: [Num Nodi, 384], si utilizza SBERT

device = torch.device('cpu')
X_tensor = X_tensor.to(device)

# Pre-calcoliamo le variabili strutturali fuori dall'objective per massima efficienza
node_to_idx = encoder.node_id_to_idx 
NUM_NODES = len(node_to_idx)
IN_CHANNELS = X_tensor.shape[1]

# ==========================================
# FASE DI MODEL SELECTION (OPTUNA)
# ==========================================
def objective(trial):
    # 1. OPTUNA SUGGERISCE GLI IPERPARAMETRI DA TESTARE
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    margin = trial.suggest_categorical('margin', [0.5, 1.0, 1.5])
    k_negatives = trial.suggest_int('k_negatives', 1, 4)

    # --- C. Genera il Dataset (Negative Sampling per TSV) ---
    tutti_i_df = []
    relazioni_globali = set()

    # Iteriamo un libro (TSV) alla volta per mantenere le falsificazioni coerenti
    for tsv_file in tsv_files:
        sampler = GraphNegativeSampler()
        sampler.load_tsv([tsv_file]) # Carica gli archi del singolo libro
        
        # Genera i falsi logici per questo libro
        df_libro = sampler.generate_dataset(k_negatives=k_negatives) 
        tutti_i_df.append(df_libro)
        
        # Aggiorna il vocabolario globale degli archi usati (es. AVVIENE_DOPO, COME)
        relazioni_globali.update(sampler.relations)

    df_dataset_strings = pd.concat(tutti_i_df, ignore_index=True)

    # --- D. Il "PONTE": Mappa stringhe -> interi per PyTorch ---
    unique_relations = list(relazioni_globali)
    rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
    NUM_RELATIONS = len(rel_to_idx)

    # Traduzione del DataFrame in indici numerici
    df_dataset = df_dataset_strings.copy()
    df_dataset['head'] = df_dataset['head'].map(node_to_idx)
    df_dataset['tail'] = df_dataset['tail'].map(node_to_idx)
    df_dataset['relation'] = df_dataset['relation'].map(rel_to_idx)
    
    # Pulizia da eventuali NaN (dovuti a mismatch di caratteri speciali)
    df_dataset = df_dataset.dropna().astype({'head': 'int64', 'tail': 'int64', 'relation': 'int64', 'label': 'int64'})

    # Archi reali (usati dalla R-GCN per il Message Passing)
    true_triplets_df = df_dataset[df_dataset['label'] == 1]

    # --- E. Inizializza e Addestra il Modello ---
    model = NarrativeKGModel(
        num_nodes=NUM_NODES, 
        num_relations=NUM_RELATIONS, 
        in_channels=IN_CHANNELS, 
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device=device)

    # Lancia l'addestramento con la configurazione di questo tentativo
    trained_model, training_history = train_kg_model(
        model=model, 
        df_dataset=df_dataset, 
        true_triplets_df=true_triplets_df, 
        node_features_X=X_tensor, 
        num_epochs=100, # Basso per velocizzare Optuna
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
    )
    
    # Optuna userà la minor Validation Loss ottenuta in queste 100 epoche
    best_val_loss = min(training_history['val_loss'])
    return best_val_loss


# ==========================================
# ESECUZIONE MAIN
# ==========================================
if __name__ == "__main__":
    print("\nAvvio Model Selection con Optuna (Versione TSV)...")
    
    # Minimizzare l'errore
    study = optuna.create_study(direction='minimize')

    # Numero di configurazioni da testare (es. 20)
    study.optimize(objective, n_trials=20) 

    print("\n=== MODEL SELECTION COMPLETATA ===")
    print(f"Miglior Validation Loss raggiunta: {study.best_value}")
    print("Iperparametri Ottimali Trovati:")
    for key, value in study.best_params.items():
        print(f"  * {key}: {value}")

# Caso 1
# Miglior Validation Loss raggiunta: 0.2253800481557846
# Iperparametri Ottimali Trovati:
#  * hidden_channels: 32
#  * num_layers: 3
#  * dropout_rate: 0.10174214482282506
#  * lr: 0.008846175564081828
#  * weight_decay: 2.6756844783872094e-05
#  * margin: 0.5
#  * k_negatives: 3

# Caso 2
# Miglior Validation Loss raggiunta: 0.2781398296356201
# Iperparametri Ottimali Trovati:
#   * hidden_channels: 64
#   * num_layers: 2
#   * dropout_rate: 0.22943336121970184
#   * lr: 0.0018238113989311065
#   * weight_decay: 0.0001164486298898722
#   * margin: 0.5
#   * k_negatives: 4