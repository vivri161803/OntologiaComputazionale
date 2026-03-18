import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

from GNN.EncoderDecoder import NarrativeKGModel
from GNN.pretraining.early_stopping import EarlyStopping
from GNN.BERT.bert_csv import NodeFeatureEncoder
from GNN.pretraining.NegativeSampling_csv import GraphNegativeSampler

class GNNTrainer:
    """
    Classe unificata per la gestione dell'addestramento della GNN 
    (R-GCN + TransE). Incapsula preparazione dati, loop di training e salvataggio.
    """
    def __init__(self, tsv_files: list, params: dict, device: torch.device = torch.device('cpu')):
        self.tsv_files = tsv_files
        self.params = params
        self.device = device
        
        self.encoder = NodeFeatureEncoder()
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}

    def prepare_data(self):
        print("\n[1] Estrazione Feature Iniziali (SBERT)...")
        self.encoder.process_tsv(self.tsv_files)
        self.X_tensor = self.encoder.generate_feature_matrix().to(self.device)
        
        print(f"\n[2] Generazione Dataset con Negative Sampling (K={self.params['k_negatives']})...")
        tutti_i_df = []
        relazioni_globali = set()

        for tsv_file in self.tsv_files:
            sampler = GraphNegativeSampler()
            sampler.load_tsv([tsv_file])
            df_libro = sampler.generate_dataset(k_negatives=self.params['k_negatives']) 
            tutti_i_df.append(df_libro)
            relazioni_globali.update(sampler.relations)

        df_dataset_strings = pd.concat(tutti_i_df, ignore_index=True)
        
        print("\n[3] Preparazione Tensori...")
        self.node_to_idx = self.encoder.node_id_to_idx 
        self.idx_to_node = self.encoder.idx_to_node_id
        
        unique_relations = list(relazioni_globali)
        self.rel_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}
        self.idx_to_rel = {v: k for k, v in self.rel_to_idx.items()}
        
        df_dataset = df_dataset_strings.copy()
        df_dataset['head'] = df_dataset['head'].map(self.node_to_idx)
        df_dataset['tail'] = df_dataset['tail'].map(self.node_to_idx)
        df_dataset['relation'] = df_dataset['relation'].map(self.rel_to_idx)
        self.df_dataset = df_dataset.dropna().astype({'head': 'int64', 'tail': 'int64', 'relation': 'int64', 'label': 'int64'})
        
        self.true_triplets_df = self.df_dataset[self.df_dataset['label'] == 1]
        
    def _df_to_tensors(self, df: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        heads = torch.tensor(df['head'].values, dtype=torch.long).to(self.device)
        rels = torch.tensor(df['relation'].values, dtype=torch.long).to(self.device)
        tails = torch.tensor(df['tail'].values, dtype=torch.long).to(self.device)
        labels = torch.tensor(df['label'].values, dtype=torch.float).to(self.device)
        return heads, rels, tails, labels
        
    def _calculate_loss(self, h, r, t, labels, node_emb, rel_emb, criterion):
        scores = self.model.score_triplets(node_emb, rel_emb, h, r, t)
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        
        num_neg = neg_scores.size(0)
        num_pos = pos_scores.size(0)
        if num_neg > 0 and num_pos > 0:
            pos_scores_matched = pos_scores.repeat_interleave(num_neg // num_pos + 1)[:num_neg]
            target = torch.ones_like(neg_scores)
            return criterion(pos_scores_matched, neg_scores, target)
        
        return torch.tensor(0.0, requires_grad=True).to(self.device) 

    def train(self, num_epochs: int = 200, save_plot: bool = True):
        print("\n[4] Inizializzazione Modello Ottimizzato...")
        self.model = NarrativeKGModel(
            num_nodes=len(self.node_to_idx), 
            num_relations=len(self.rel_to_idx), 
            in_channels=self.X_tensor.shape[1], 
            hidden_channels=self.params["hidden_channels"],
            num_layers=self.params["num_layers"],
            dropout_rate=self.params["dropout_rate"]
        ).to(self.device)
        
        edge_index = torch.tensor([
            self.true_triplets_df['head'].values, 
            self.true_triplets_df['tail'].values
        ], dtype=torch.long).to(self.device)
        edge_type = torch.tensor(self.true_triplets_df['relation'].values, dtype=torch.long).to(self.device)
        
        df_train, df_val = train_test_split(self.df_dataset, test_size=0.2, random_state=42)
        train_h, train_r, train_t, train_labels = self._df_to_tensors(df_train)
        val_h, val_r, val_t, val_labels = self._df_to_tensors(df_val)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        criterion = nn.MarginRankingLoss(margin=self.params["margin"])
        early_stopping = EarlyStopping(patience=15)
        
        print("\n[5] Avvio Addestramento Definitivo...")
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            
            node_emb, rel_emb = self.model(self.X_tensor, edge_index, edge_type)
            loss_train = self._calculate_loss(train_h, train_r, train_t, train_labels, node_emb, rel_emb, criterion)
            
            loss_train.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                node_emb_val, rel_emb_val = self.model(self.X_tensor, edge_index, edge_type)
                loss_val = self._calculate_loss(val_h, val_r, val_t, val_labels, node_emb_val, rel_emb_val, criterion)
                
            self.history['train_loss'].append(loss_train.item())
            self.history['val_loss'].append(loss_val.item())
                
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f}")
            
            early_stopping(loss_val.item(), self.model)
            if early_stopping.early_stop:
                print("Early stopping innescato! Interruzione dell'addestramento.")
                break
                
        early_stopping.restore_best_weights(self.model)
        
        if save_plot:
            self.plot_history()
            
    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss', color='blue', linewidth=2)
        plt.plot(self.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title('Andamento Loss Finale (Modello Ottimizzato)', fontsize=14)
        plt.xlabel('Epoche', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('final_training_plot.png', dpi=300, bbox_inches='tight')

    def save_model(self, out_dir: str = "TrainedModel"):
        print("\n[6] Estrazione e Salvataggio Embedding Finali...")
        self.model.eval()
        with torch.no_grad():
            edge_index = torch.tensor([self.true_triplets_df['head'].values, self.true_triplets_df['tail'].values], dtype=torch.long).to(self.device)
            edge_type = torch.tensor(self.true_triplets_df['relation'].values, dtype=torch.long).to(self.device)
            final_node_embeddings, final_rel_embeddings = self.model(self.X_tensor, edge_index, edge_type)

        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f'{out_dir}/narrative_kg_model_weights.pt')
        torch.save(final_node_embeddings, f'{out_dir}/final_node_embeddings.pt')
        torch.save(final_rel_embeddings, f'{out_dir}/final_rel_embeddings.pt')

        with open(f'{out_dir}/node_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.idx_to_node, f, indent=4)
        with open(f'{out_dir}/relation_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.idx_to_rel, f, indent=4)

        print("\n=== PIPELINE COMPLETATA CON SUCCESSO! ===")
        print(f"Dimensione Vettoriale dello Spazio Latente: {final_node_embeddings.shape[1]}")

if __name__ == "__main__":
    from train_set import train_library, train_params_3
    trainer = GNNTrainer(tsv_files=train_library, params=train_params_3)
    trainer.prepare_data()
    trainer.train(num_epochs=500)
    trainer.save_model()
