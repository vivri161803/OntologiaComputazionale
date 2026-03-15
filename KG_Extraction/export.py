import csv
import networkx as nx

def esporta_per_embedding(grafo: nx.DiGraph, percorso_output: str):
    """
    Schiaccia le entità testuali ed esporta solo la struttura topologica (ID e Valori Chiusi)
    in un formato TSV (Tab-Separated Values) perfetto per gli algoritmi KGE.
    """
    with open(percorso_output, 'w', encoding='utf-8', newline='') as f:
        # Usiamo il tab come delimitatore, standard per i dataset di KGE
        writer = csv.writer(f, delimiter='\t')
        
        # Intestazione (Soggetto, Relazione, Oggetto)
        writer.writerow(['head', 'relation', 'tail'])
        
        # Iteriamo su tutti gli archi del grafo globale
        for source, target, data in grafo.edges(data=True):
            relazione = data['label']
            
            # Scriviamo la tripletta pura nel file
            writer.writerow([source, relazione, target])
            
    print(f"Esportazione TSV completata! File pronto per l'embedding: {percorso_output}")

