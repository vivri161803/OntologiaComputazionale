import optuna
import torch
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from KG_Extraction.books import * 
from GNN.BERT.bert import NodeFeatureEncoder
from GNN.pretraining.NegativeSampling import GraphNegativeSampler
from GNN.EncoderDecoder import NarrativeKGModel
from GNN.pretraining.train_loop import train_kg_model
import matplotlib.pyplot as plt

# --- A. Definisci i tuoi file JSON ---
biblioteca = [
    TheVerdict,   
    MobyDick, 
    Jekyll_Hyde
]
tutti_i_json = [json_file for libro in biblioteca for json_file in libro]

# --- B. Estrai i vettori di testo (SBERT) una volta sola ---
encoder = NodeFeatureEncoder()
encoder.process_chunks(tutti_i_json)
X_tensor = encoder.generate_feature_matrix() # Shape: [Num Nodi, 384]

device = torch.device('cpu')
X_tensor = X_tensor.to(device)

# Pre-calcoliamo queste variabili fuori dall'objective per efficienza
node_to_idx = encoder.node_id_to_idx 
NUM_NODES = len(node_to_idx)
IN_CHANNELS = X_tensor.shape[1]

# ==========================================
# FASE DI MODEL SELECTION (OPTUNA)
# ==========================================
def objective(trial):
    # 1. OPTUNA SUGGERISCE GLI IPERPARAMETRI DA TESTARE
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    margin = trial.suggest_categorical('margin', [0.5, 1.0, 1.5])
    k_negatives = trial.suggest_int('k_negatives', 1, 4)

    # --- C. Genera il Dataset (Negative Sampling) ---
    tutti_i_df = []
    relazioni_globali = {"AVVIENE_IN", "AVVIENE_DOPO", "COME", "PROVIENE_DA"}

    for libro_chunks in biblioteca:
        sampler = GraphNegativeSampler()
        sampler.load_chunks(libro_chunks)
        
        # Usiamo il k_negatives suggerito da Optuna
        df_libro = sampler.generate_dataset(k_negatives=k_negatives) 
        tutti_i_df.append(df_libro)
        relazioni_globali.update(sampler.partecipa_come_relations)

    df_dataset_strings = pd.concat(tutti_i_df, ignore_index=True)

    # --- D. Il "PONTE": Mappa stringhe -> interi per PyTorch ---
    unique_relations = list(relazioni_globali)
    rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
    NUM_RELATIONS = len(rel_to_idx)

    df_dataset = df_dataset_strings.copy()
    df_dataset['head'] = df_dataset['head'].map(node_to_idx)
    df_dataset['tail'] = df_dataset['tail'].map(node_to_idx)
    df_dataset['relation'] = df_dataset['relation'].map(rel_to_idx)
    df_dataset = df_dataset.dropna().astype({'head': 'int64', 'tail': 'int64', 'relation': 'int64', 'label': 'int64'})

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

    # Passiamo gli iperparametri al ciclo di training
    trained_model, training_history = train_kg_model(
        model=model, 
        df_dataset=df_dataset, 
        true_triplets_df=true_triplets_df, 
        node_features_X=X_tensor, 
        num_epochs=100, # Per velocizzare la Model Selection abbassiamo le epoche a 100
        lr=lr,
        weight_decay=weight_decay,
        margin=margin
    )
    
    # Optuna valuta il modello basandosi sulla Loss di Validazione più bassa ottenuta
    best_val_loss = min(training_history['val_loss'])
    return best_val_loss


print("\nAvvio Model Selection con Optuna...")
# Creiamo uno studio per "minimizzare" la Validation Loss
study = optuna.create_study(direction='minimize')

# Eseguiamo 20 tentativi esplorativi (modifica n_trials per testarne di più/meno)
study.optimize(objective, n_trials=20) 

print("\n=== MODEL SELECTION COMPLETATA ===")
print(f"Miglior Validation Loss raggiunta: {study.best_value}")
print("Iperparametri Ottimali Trovati:")
for key, value in study.best_params.items():
    print(f"  * {key}: {value}")

#Miglior Validation Loss raggiunta: 0.30003300309181213
#Iperparametri Ottimali Trovati:
#  * hidden_channels: 64
#  * num_layers: 2
#  * dropout_rate: 0.20964936029213044
#  * lr: 0.0024225080521569266
#  * weight_decay: 0.004000507508894926
#  * margin: 0.5
#  * k_negatives: 3