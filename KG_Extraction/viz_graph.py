import pandas as pd
import networkx as nx
from pyvis.network import Network

print("Generazione del grafo interattivo a colori in corso...")
path_file_to_read = "book/Animal_Farm/Animal_Farm.tsv"
name_output_file = "AnimalFarm.html"

# 1. Leggi il TSV
df = pd.read_csv(path_file_to_read, sep="\t")

# Attributi per gli archi (frecce)
df['label'] = df['relation']
df['title'] = df['relation']

# 2. Crea il grafo di base
G = nx.from_pandas_edgelist(
    df, 
    source='head', 
    target='tail', 
    edge_attr=['label', 'title'],
    create_using=nx.DiGraph()
)

# 3. IL TRUCCO DEI COLORI E DELLE FORME: Analizziamo i prefissi dei nodi
for node in G.nodes():
    node_str = str(node)
    
    # Assegniamo l'etichetta visibile al nodo (il suo ID)
    G.nodes[node]["label"] = node_str
    G.nodes[node]["title"] = f"Tipologia/ID: {node_str}"
    
    # Personaggi (Raw Text)
    if node_str.startswith("char_"):
        G.nodes[node]["color"] = "#ff6666" # Rosso pastello
        G.nodes[node]["shape"] = "dot"
        
    # Eventi (Raw Text)
    elif node_str.startswith("evt_") and "type" not in node_str:
        G.nodes[node]["color"] = "#66b3ff" # Azzurro
        G.nodes[node]["shape"] = "square"
        
    # Luoghi (Raw Text)
    elif node_str.startswith("loc_") and "type" not in node_str:
        G.nodes[node]["color"] = "#99ff99" # Verde chiaro
        G.nodes[node]["shape"] = "triangle"
        
    # Oggetti o altro
    elif node_str.startswith("obj_"):
        G.nodes[node]["color"] = "#ffcc99" # Arancione
        G.nodes[node]["shape"] = "star"
        
    # VALORI CHIUSI (Ontologia universale: role_, evt_type_, loc_type_)
    elif "role_" in node_str or "type_" in node_str:
        G.nodes[node]["color"] = "#ffd700" # Oro brillante per i nodi strutturali
        G.nodes[node]["shape"] = "diamond"
        # Ingrandiamo i valori chiusi perché sono i nodi centrali su cui convergono i dati
        G.nodes[node]["size"] = 25 
        
    # Fallback per nodi non riconosciuti
    else:
        G.nodes[node]["color"] = "#cccccc" # Grigio

# 4. Crea la rete Pyvis
# Aggiungiamo select_menu e filter_menu per avere una UI di ricerca nel browser!
net = Network(height='800px', width='100%', bgcolor='#1a1a1a', font_color='white', directed=True, select_menu=True, filter_menu=True)

# Spaziatura dinamica per evitare grovigli
net.force_atlas_2based(gravity=-50, spring_length=150, overlap=1)

# 5. Importa i dati e genera l'HTML
net.from_nx(G)
file_html = name_output_file
net.show(file_html, notebook=False)

print(f"Fatto! Apri il file '{file_html}' nel tuo browser.")