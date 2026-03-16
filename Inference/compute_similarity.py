import pandas as pd
import torch
import torch.nn.functional as F
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from saveplot import save_plot

csv_path = "Inference/csv/Test.csv"
name_sum_hm = "Inference/plot/1similarity_heatmap_sum.png"
name_concat_hm = "Inference/plot/1similarity_heatmap_concat.png"

# 1. Carica il CSV
df = pd.read_csv(csv_path)
print("Dati caricati. Shape:", df.shape)

# 2. RISOLUZIONE TRAPPOLA CSV: Convertiamo le stringhe in liste vere e proprie
# ast.literal_eval legge la stringa "[0.1, 0.2]" e la trasforma in una lista Python [0.1, 0.2]
df["ConcatenationEmbeddings"] = df["ConcatenationEmbeddings"].apply(ast.literal_eval)
df["SumEmbeddings"] = df["SumEmbeddings"].apply(ast.literal_eval)

# 3. Trasformiamo le liste in tensori PyTorch 2D (Shape: [Num_Libri, Dim_Embedding])
concat_tens = torch.tensor(df["ConcatenationEmbeddings"].tolist(), dtype=torch.float32)
sum_tens = torch.tensor(df["SumEmbeddings"].tolist(), dtype=torch.float32)
titoli = df["Titolo"]

# 4. CALCOLO DELLA SIMILARITÀ (Tutti contro Tutti)
# Il modo matematicamente più veloce per fare Cosine Similarity tra tutti gli elementi 
# è normalizzare i vettori (L2 norm) e poi fare la moltiplicazione matriciale.
concat_norm = F.normalize(concat_tens, p=2, dim=1)
sim_concat_matrix = torch.mm(concat_norm, concat_norm.T) # .mm fa il prodotto riga per colonna

sum_norm = F.normalize(sum_tens, p=2, dim=1)
sim_sum_matrix = torch.mm(sum_norm, sum_norm.T)

print("\nMatrice di Similarità (Concatenation) creata! Shape:", sim_concat_matrix.shape)
print("")
print(sim_concat_matrix)
print("")
print(sim_sum_matrix)
# La diagonale sarà tutta composta da 1.0 (perché ogni libro è identico a se stesso).

# Generazione del plot (heatmap delle similarita')
save_plot(sim_sum_matrix, titoli, 'Similarità Strutturale dei Romanzi (Sum Pooling)', name_sum_hm)
save_plot(sim_concat_matrix, titoli, 'Similarità Strutturale dei Romanzi (Concat Pooling)', name_concat_hm)