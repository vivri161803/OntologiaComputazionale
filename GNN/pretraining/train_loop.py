import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from GNN.pretraining.early_stopping import EarlyStopping

def train_kg_model(model, df_dataset, true_triplets_df, node_features_X, num_epochs=200, lr=0.005, weight_decay=1e-4, margin=1.0):
    """
    Esegue il training della GNN con Early Stopping.
    
    df_dataset: Il DataFrame contenente triplette vere (label 1) e false (label 0): head-relation-tail-label
    true_triplets_df: DataFrame contenente SOLO le triplette vere (per il message passing).
    node_features_X: Il tensore generato da SBERT (Shape: [Num_Nodi, 384]).
    """
    
    # Recuperiamo il device dalla matrice SBERT passata in input
    device = node_features_X.device
    
    # 1. PREPARAZIONE DEL GRAFO STRUTTURALE (Solo archi veri)
    # Aggiungiamo .to(device) per allineare i tensori al device corretto
    edge_index = torch.tensor([
        true_triplets_df['head'].values, 
        true_triplets_df['tail'].values
    ], dtype=torch.long).to(device)
    
    edge_type = torch.tensor(true_triplets_df['relation'].values, dtype=torch.long).to(device)
    
    # 2. SPLIT DEL DATASET DI VALUTAZIONE (Train / Validation)
    df_train, df_val = train_test_split(df_dataset, test_size=0.2, random_state=42)
    
    def df_to_tensors(df):
        # Spostiamo anche tutti i tensori del dataset sul device corretto
        heads = torch.tensor(df['head'].values, dtype=torch.long).to(device)
        rels = torch.tensor(df['relation'].values, dtype=torch.long).to(device)
        tails = torch.tensor(df['tail'].values, dtype=torch.long).to(device)
        labels = torch.tensor(df['label'].values, dtype=torch.float).to(device)
        return heads, rels, tails, labels

    train_h, train_r, train_t, train_labels = df_to_tensors(df_train)
    val_h, val_r, val_t, val_labels = df_to_tensors(df_val)
    
    # 3. SETUP OTTIMIZZATORE E LOSS
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=margin)
    early_stopping = EarlyStopping(patience=15)
    
    # Funzione di supporto per calcolare la loss su un set di triplette
    def calculate_loss(h, r, t, labels, node_emb, rel_emb):
        # TransE scoring: h+r-t
        scores = model.score_triplets(node_emb, rel_emb, h, r, t)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        
        num_neg = neg_scores.size(0)
        num_pos = pos_scores.size(0)
        if num_neg > 0 and num_pos > 0:
            pos_scores_matched = pos_scores.repeat_interleave(num_neg // num_pos + 1)[:num_neg]
            target = torch.ones_like(neg_scores) # ones_like eredita automaticamente il device
            return criterion(pos_scores_matched, neg_scores, target)
        
        # Fallback in caso di batch anomali, forzato sul device
        return torch.tensor(0.0, requires_grad=True).to(device) 

    history = {'train_loss': [], 'val_loss': []}

    # 4. TRAINING LOOP
    print("Inizio Addestramento...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # encoder: R-CGN
        node_emb, rel_emb = model(node_features_X, edge_index, edge_type)
        loss_train = calculate_loss(train_h, train_r, train_t, train_labels, node_emb, rel_emb)
        
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            node_emb_val, rel_emb_val = model(node_features_X, edge_index, edge_type)
            loss_val = calculate_loss(val_h, val_r, val_t, val_labels, node_emb_val, rel_emb_val)
            
        # --- NUOVO: Salviamo i valori nell'history ---
        history['train_loss'].append(loss_train.item())
        history['val_loss'].append(loss_val.item())
            
        print(f"Epoch {epoch+1:03d} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f}")
        
        early_stopping(loss_val.item(), model)
        if early_stopping.early_stop:
            print("Early stopping innescato! Interruzione dell'addestramento.")
            break
            
    early_stopping.restore_best_weights(model)
    
    # --- NUOVO: Restituiamo anche la history ---
    return model, history