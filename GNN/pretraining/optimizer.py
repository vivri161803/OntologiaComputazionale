import optuna
import torch
import pandas as pd
from typing import List

from GNN.pretraining.trainer import GNNTrainer

class GNNHyperparameterOptimizer:
    """
    Gestisce la Model Selection tramite libreria Optuna 
    per esplorare lo spazio degli iperparametri della GNN.
    """
    def __init__(self, tsv_files: List[str]):
        self.tsv_files = tsv_files
        
    def objective(self, trial):
        params = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'margin': trial.suggest_categorical('margin', [0.5, 1.0, 1.5]),
            'k_negatives': trial.suggest_int('k_negatives', 1, 4)
        }
        
        # Sfruttiamo il GNNTrainer OOP
        trainer = GNNTrainer(tsv_files=self.tsv_files, params=params)
        trainer.prepare_data()
        trainer.train(num_epochs=100, save_plot=False)
        
        best_val_loss = min(trainer.history['val_loss'])
        return best_val_loss

    def optimize(self, n_trials: int = 20):
        print("\nAvvio Model Selection con Optuna (Versione TSV OOP)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials) 
        
        print("\n=== MODEL SELECTION COMPLETATA ===")
        print(f"Miglior Validation Loss raggiunta: {study.best_value}")
        print("Iperparametri Ottimali Trovati:")
        for key, value in study.best_params.items():
            print(f"  * {key}: {value}")
        return study.best_params

if __name__ == "__main__":
    from train_set import train_library
    optimizer = GNNHyperparameterOptimizer(tsv_files=train_library)
    optimizer.optimize(n_trials=20)
