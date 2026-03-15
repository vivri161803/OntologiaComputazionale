import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Quante epoche aspettare senza miglioramenti prima di fermarsi.
            min_delta: Miglioramento minimo per essere considerato valido.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            # Miglioramento! Salviamo i pesi
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()
        else:
            # Nessun miglioramento
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model: torch.nn.Module):
        """Ripristina i pesi dell'epoca con la loss minore."""
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            print("Pesi del modello ripristinati all'epoca migliore.")