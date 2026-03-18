import csv
import networkx as nx

class GraphTSVExporter:
    """
    Gestisce l'esportazione del grafo NetworkX in un formato relazionale
    TSV (Head, Relation, Tail) essenziale per gli algoritmi KGE.
    """
    def export(self, graph: nx.DiGraph, output_path: str):
        """Schiaccia le entità testuali ed esporta la struttura topologica."""
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['head', 'relation', 'tail'])
            
            for source, target, data in graph.edges(data=True):
                relation = data.get('label', '')
                writer.writerow([source, relation, target])
                
        print(f"Esportazione TSV completata! File pronto: {output_path}")
