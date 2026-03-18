# Computazionale Ontology - Python Modules Overview

Questo repository contiene il codice sorgente per l'estrazione, l'elaborazione, l'addestramento e l'inferenza di Knowledge Graph narrativi a partire da testi letterari.

Di seguito è presentata una panoramica esclusiva dei file `.py` presenti all'interno del progetto, suddivisi per aree logiche di competenza. I file `.json` e `.csv` o `.tsv` sono omessi, come richiesto.

## 1. Estrazione del Knowledge Graph e Ontologia
Questi script gestiscono il processo di trasformazione del testo grezzo del romanzo in un Knowledge Graph direzionato, sfruttando LLM (Claude) e una rigorosa ontologia predefinita.

* **`KG_from_txt.py`**: È il main script che orchestra l'intera pipeline di estrazione testuale, dal file `.txt` grezzo fino all'esportazione in file testuali strutturati, passando per le tre fasi principali (chunking, entity resolution, export).
* **`onthology/onthology.py`**: Definisce rigorosamente le classi (basate su Pydantic ed Enum) che costituiscono l'Ontologia del Knowledge Graph, includendo le entità a Vocabolario Aperto (Personaggi, Luoghi, Eventi) e i nodi a Valore Chiuso (Ruoli archetipici, Fasi della trama).
* **`KG_Extraction/micro.py`**: Fornisce le funzioni per leggere un libro, dividerlo in frammenti (chunking) coerenti e chiamare l'API di Anthropic per estrarre porzioni di sub-grafo locale (nodi e triplette) per ciascun frammento.
* **`KG_Extraction/macro.py`**: Combina i vari sub-grafi estratti in un unico grafo generale. Utilizza LLM per fare "Entity Resolution" in batch (unificando ID simili, come 'char_strider' e 'char_aragorn') e costruisce logicamente gli archi temporali.
* **`KG_Extraction/export.py`**: Esporta la topologia del NetworkX Graph completo in un formato TSV classico (Head, Relation, Tail) essenziale per gli algoritmi di addestramento vettoriale.
* **`KG_Extraction/viz_graph.py`**: Genera una mappa HTML interattiva e colorata del grafo completo a partire dal dataset estratto, differenziando visivamente per colore e forma i vari componenti narrativi.
* **`KG_Extraction/books.py`**: Contiene delle liste di configurazione che raggruppano i percorsi dei chunk estratti in formato JSON per le varie opere (es. Moby Dick, Jekyll & Hyde).
* **`KG_Extraction/api_key.py`**: File di impostazione contenente le chiavi API di Anthropic utilizzate nell'estrazione.
* **`book/get_txt.py`**: Script per il prelievo automatizzato ("scraping") del testo integrale dei grandi classici letterari direttamente dal Project Gutenberg Australia.

## 2. Graph Neural Network (GNN) e Pretraining
Script per tradurre la rete narrativa estratta in Embeddings vettoriali (Knowledge Graph Embeddings) tramite architetture di Deep Learning basate su GNN e Modelli Linguistici (SBERT).

* **`pretraining_tsv.py`**: Script principale e definitivo per l'addestramento end-to-end del modello GNN, operante sui dati in formato TSV. Produce e salva i pesi della rete neurale ottimizzati e le rappresentazioni latenti.
* **`GNN/EncoderDecoder.py`**: Implementa architettura del modello ibrido `NarrativeKGModel` in PyTorch, configurata con un Encoder R-GCN (Relational Graph Convolutional Network) per il Message Passing e un Decoder TransE.
* **`GNN/BERT/bert.py`** e **`GNN/BERT/bert_csv.py`**: Svolgono il ruolo di codifica semantica testuale. Utilizzano il modello specializzato SBERT (`all-MiniLM-L6-v2`) per convertire il significato testuale dei nodi (da JSON o TSV) all'interno di una matrice di input vettoriale.
* **`GNN/pretraining/NegativeSampling.py`** e **`GNN/pretraining/NegativeSampling_csv.py`**: Meccanismi di generazione procedurale intelligente dei "falsi" (corrupted triplets). Alterano strategicamente Teste, Code o Relazioni (da formati logici diversi) per insegnare al modello a distinguere archi validi da archi insensati.
* **`GNN/pretraining/train_loop.py`**: Contiene la logica base del ciclo di addestramento (`train_kg_model`), con la Margin Ranking Loss, il propagation feed forward e backward e le routine di salvataggio dell'history.
* **`GNN/pretraining/early_stopping.py`**: Provvede al modulo di arresto anticipato (Early Stopping) per monitorare la Loss di validazione, impedendo l'overfitting e salvando in memoria i pesi storici migliori.
* **`optuna_pretraining_tsv.py`**: Applica la libreria Optuna per esplorare in modo efficiente lo spazio degli iperparametri (lr, hidden channels, num layers, dropout, ecc.) sul task TSV e selezionare la combinazione migliore per la GNN.
* **`train_set.py`**: Un modulo di configurazione che esporta come costanti le liste d'addestramento e iterazione dei vari romanzi, nonché i risultati migliori registrati via Optuna da riutilizzare ai fini del Training e dell'Inferenza.

## 3. Inferenza e Rappresentazione (Visiva e Analitica)
Programmi dedicati al test del modello addestrato su nuovi dati testuali inediti, al calcolo di fingerprint metriche e alla generazione di plot dimensionali.

* **`inference_pooling.py`**: Estrae una rappresentazione vettoriale sintetica complessiva ("Graph Embedding" o impronta digitale semantica) per uno specifico romanzo inedito. Calcola poolings avanzati sui vettori nodali calcolati in inferenza (Mean Pooling, Max, Sum, Attention) concatenandoli.
* **`Inference/compute_similarity.py`**: Ingerisce i Graph Embeddings risultanti post-pooling ed esegue moltiplicazioni vettoriali cosine-based per determinare e comparare la distanza/similarità strutturale tra di essi.
* **`Inference/similarity_heatmap.py`**: Contiene la funzione helper `hm` per renderizzare, a partire dalla matrice matematica delle similarità tra le opere, un'elegante Heatmap di colore termico avvalendosi di Seaborn.
* **`tsne_plot_embedding.py`**: Utilizza T-SNE con Plotly per ridurre lo spazio latente generato dai tensori a 2 dimensioni esplorabili in una pagina web. Mantiene il tracking delle categorie logiche d'appartenenza per la colorazione puntiforme visiva.
* **`umap_plot_embedding.py`**: Riduce allo stesso modo le dimensioni complesse degli spazi vettoriali dei personaggi a 2D avvalendosi del più recente algoritmo UMAP, utile a rivelature geometrie e cluster testuali più ravvicinati a livello macroscopico.
