import pandas as pd
import torch
import torch.nn.functional as F
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from similarity_heatmap import hm

csv_path = "Inference/csv/Total.csv"
name_sum_hm = "Inference/plot/Total_hm_sum.png" # sommando tutti i pooling ottenuti per il romanzo singolo 
name_concat_hm = "Inference/plot/Total_hm_concat.png" # concantenando i pooling ottenuti, uno per ogni romanzo
name_attention_hm = "Inference/plot/Total_hm_attention.png" # utilizzando solamente il pooling calcolato con attention

# 1. Carica il CSV
df = pd.read_csv(csv_path)
print("Dati caricati. Shape:", df.shape)

# 2. RISOLUZIONE TRAPPOLA CSV: Convertiamo le stringhe in liste vere e proprie
# ast.literal_eval legge la stringa "[0.1, 0.2]" e la trasforma in una lista Python [0.1, 0.2]
df["ConcatenationEmbeddings"] = df["ConcatenationEmbeddings"].apply(ast.literal_eval)
df["SumEmbeddings"] = df["SumEmbeddings"].apply(ast.literal_eval)
df["AttentionEmbedding"] = df["AttentionEmbedding"].apply(ast.literal_eval)

# 3. Trasformiamo le liste in tensori PyTorch 2D (Shape: [Num_Libri, Dim_Embedding])
concat_tens = torch.tensor(df["ConcatenationEmbeddings"].tolist(), dtype=torch.float32)
sum_tens = torch.tensor(df["SumEmbeddings"].tolist(), dtype=torch.float32)
attention_tens = torch.tensor(df["AttentionEmbedding"].to_list(), dtype=torch.float32)
titoli = df["Titolo"]

# 4. CALCOLO DELLA SIMILARITÀ (Tutti contro Tutti)
# Il modo matematicamente più veloce per fare Cosine Similarity tra tutti gli elementi 
# è normalizzare i vettori (L2 norm) e poi fare la moltiplicazione matriciale.

# max, mean, attention, sum concatenati
concat_norm = F.normalize(concat_tens, p=2, dim=1)
sim_concat_matrix = torch.mm(concat_norm, concat_norm.T) # .mm fa il prodotto riga per colonna

# max, mean, attention, sum pooling sommati 
sum_norm = F.normalize(sum_tens, p=2, dim=1)
sim_sum_matrix = torch.mm(sum_norm, sum_norm.T)

# solamente attention pooling 
attention_norm = F.normalize(attention_tens, p=2, dim = 1)
sim_attention_matrix = torch.mm(attention_norm, attention_norm.T) 

print("\nMatrice di Similarità (Concatenation) creata! Shape:", sim_concat_matrix.shape)
print("")
print(sim_concat_matrix)
print("")
print(sim_sum_matrix)
print("")
print(sim_attention_matrix)
# La diagonale sarà tutta composta da 1.0 (perché ogni libro è identico a se stesso).

# Generazione del plot (heatmap delle similarita')
hm(sim_sum_matrix, titoli, 'Similarità Strutturale dei Romanzi (Sum Pooling)', name_sum_hm)
hm(sim_concat_matrix, titoli, 'Similarità Strutturale dei Romanzi (Concat Pooling)', name_concat_hm)
hm(sim_attention_matrix, titoli, 'Similarità Strutturale dei Romanzi (Attention Pooling)', name_attention_hm)