import matplotlib.pyplot as plt
import seaborn as sns

def hm(matrix, titoli, title, output_name):
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        matrix, 
        annot=True,              # Mostra i valori numerici dentro le celle
        fmt=".3f",               # Formatta i numeri a 3 cifre decimali
        cmap='coolwarm',         # Usa una scala di colori da blu (basso) a rosso (alto)
        xticklabels=titoli,      # Etichette asse X (Titoli dei libri)
        yticklabels=titoli,      # Etichette asse Y (Titoli dei libri)
        vmin=0.75, vmax=1.00     # Scala i colori per risaltare le differenze (i valori sono alti)
    )

    # Estetica del grafico
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12) # Inclina le etichette X per leggerle meglio
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()

    # Salvataggio dell'immagine
    plt.savefig(output_name, dpi=300)
    print("Heatmap generata e salvata con successo")