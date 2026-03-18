import pandas as pd
import torch
import torch.nn.functional as F
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class SemanticSimilarityAnalyzer:
    """
    Analizza i vettori di Graph Pooling risultanti calcolando la Cosine Similarity
    e renderizza le heatmap di similarità narratologica tra romanzi.
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        print("Dati strutturali caricati. Shape:", self.df.shape)
        
        # Pydantic automatic type ast eval
        for col in ["ConcatenationEmbeddings", "SumEmbeddings", "AttentionEmbedding"]:
            self.df[col] = self.df[col].apply(ast.literal_eval)
            
        self.concat_tens = torch.tensor(self.df["ConcatenationEmbeddings"].tolist(), dtype=torch.float32)
        self.sum_tens = torch.tensor(self.df["SumEmbeddings"].tolist(), dtype=torch.float32)
        self.attention_tens = torch.tensor(self.df["AttentionEmbedding"].to_list(), dtype=torch.float32)
        self.titles = self.df["Titolo"].tolist()
        
    def compute_similarity_matrices(self):
        """Calcola il prodotto scalare normalizzato (Cosine Similarity) su tutti asse del dataset."""
        c_norm = F.normalize(self.concat_tens, p=2, dim=1)
        self.sim_concat = torch.mm(c_norm, c_norm.T)
        
        s_norm = F.normalize(self.sum_tens, p=2, dim=1)
        self.sim_sum = torch.mm(s_norm, s_norm.T)
        
        a_norm = F.normalize(self.attention_tens, p=2, dim=1)
        self.sim_attention = torch.mm(a_norm, a_norm.T)
        
    def plot_heatmap(self, matrix: torch.Tensor, title: str, output_name: str):
        """Genera e salva la heatmap stile seaborn formattata."""
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            matrix.numpy(), annot=True, fmt=".3f", cmap='coolwarm',
            xticklabels=self.titles, yticklabels=self.titles, vmin=0.75, vmax=1.00
        )
        plt.title(title, fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        plt.savefig(output_name, dpi=300)
        print(f"Heatmap '{title}' salvata in {output_name}")
        plt.close()
        
    def generate_all_plots(self, output_dir: str = "Inference/plot"):
        self.compute_similarity_matrices()
        self.plot_heatmap(self.sim_sum, 'Similarità Strutturale (Sum Pooling)', f"{output_dir}/Total_hm_sum.png")
        self.plot_heatmap(self.sim_concat, 'Similarità Strutturale (Concat Pooling)', f"{output_dir}/Total_hm_concat.png")
        self.plot_heatmap(self.sim_attention, 'Similarità Strutturale (Attention Pooling)', f"{output_dir}/Total_hm_attention.png")

if __name__ == "__main__":
    analyzer = SemanticSimilarityAnalyzer("Inference/csv/Total.csv")
    analyzer.generate_all_plots()
