from torch_geometric.nn import RGCNConv
import torch.nn as nn
import torch 

class NarrativeKGModel(nn.Module):
    # Aggiunti num_layers e dropout_rate
    def __init__(self, num_nodes: int, num_relations: int, in_channels: int, hidden_channels: int, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        
        # embedding delle relazioni
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations, hidden_channels))
        nn.init.xavier_uniform_(self.rel_emb)
        
        # Creazione dinamica dei layer RGCN basata su num_layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        # R-GCN as encoder: works on nodes embeddings
        # Passaggio sequenziale attraverso i layer dinamici, permette di calcolare i nuovi embedding per i nodi 
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            # Applichiamo ReLU e Dropout solo tra i layer, non alla fine
            if i < len(self.convs) - 1:
                x = self.dropout(self.relu(x))
                
        return x, self.rel_emb # OUTPUT: embedding nodi e relazioni

    def score_triplets(self, node_emb, rel_emb, head_indices, rel_indices, tail_indices):
        # TransE as Decoder: works on both nodes and arcs embeddings
        # Serve per calcolare la loss durante la fase di pretraining del modello 
        h = node_emb[head_indices]
        r = rel_emb[rel_indices]
        t = node_emb[tail_indices]
        return -torch.norm(h + r - t, p=1, dim=1)