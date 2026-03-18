# Ontologia Computazionale - OOP Architecture

Questo progetto estrae Knowledge Graph da testi narrativi usando LLM (Claude Haiku), sfrutta le Graph Neural Network (R-GCN + TransE) per elaborare Graph Embeddings, e performa analisi dimensionali (UMAP/T-SNE) per calcolare similarità strutturali.

> **Nota:** L'intera architettura è stata recentemente riscritta seguendo rigidi paradigmi di **Object-Oriented Programming (OOP)** per garantire modularità, rimpiazzando i numerosi script procedurali con una pipeline a oggetti.

---

## Struttura e Moduli

Il repository è diviso in **3 package principali**, ciascuno responsabile di una fase logica:

### 1. Modulo: Acquisizione Dati ed Estrazione (KG_Extraction)
Gestisce il processo end-to-end, scaricando il testo e trasformando la narrazione in un grafo direzionato esportabile. 

- `book/book_downloader.py` (Classe **`BookDownloader`**)
  Sostituisce il vecchio script `get_txt.py`. Scarica i testi integrali dei romanzi da URL specifici e li archivia. 
- `KG_Extraction/pipeline.py` (Classe **`NarrativeKGPipeline`**)
  L'orchestratore principale che unisce estrattore, resolver ed esportatore in un singolo flusso isolato. Sostituisce `KG_from_txt.py`.
- `KG_Extraction/extractor.py` (Classe **`TextChunkExtractor`**)
  Gestisce la frammentazione del libro in test chunks e l'invocazione di *Claude* tramite `instructor` per generare sub-grafi Pydantic aderenti all'ontologia formale.
- `KG_Extraction/merger.py` (Classe **`GraphMerger`**)
  Si occupa di fondere i sub-grafi e applicare la *Entity Resolution* IA in scaglioni per unificare i nodi logici (es. "char_strider" e "char_aragorn"). Crea il NetworkX `DiGraph` globale.
- `KG_Extraction/exporter.py` (Classe **`GraphTSVExporter`**)
  Esporta il NetworkX Graph globale nel formato TSV vettoriale standard (`head`, `relation`, `tail`).
- `KG_Extraction/interactive_visualizer.py` (Classe **`PyVisGraphVisualizer`**)
  Una classe utilitaria per renderizzare HTML esplorabili della rete, codificando autonomamente per colore i nodi a seconda della tipologia narratologica estratta.

---

### 2. Modulo: Graph Neural Network & Pretraining (GNN)
Gestisce il setup, il campionamento negativo e il training loop della rete GNN personalizzata (R-GCN ed embeddings SBERT).

- `GNN/pretraining/trainer.py` (Classe **`GNNTrainer`**)
  Una possente classe unificata che integra generazione del dataset testuale SBERT, generazione di negatività (Negative Sampling) e tutto il loop di addestramento PyTorch completo con *Early Stopping*. 
- `GNN/pretraining/optimizer.py` (Classe **`GNNHyperparameterOptimizer`**)
  Incapsula `Optuna` per automatizzare la model-selection, istanziando dinamicamente plurimi `GNNTrainer` alla ricerca della Validation Loss minima prima di finalizzare l'allenamento.
- `GNN/EncoderDecoder.py` (Classe `NarrativeKGModel`)
  Ospita la definizione dell'architettura neurale PyTorch (layer convoluzionali RGCN e decoder di loss basato su TransE).
- `GNN/BERT/bert_csv.py` (Classe `NodeFeatureEncoder`) e `GNN/pretraining/NegativeSampling_csv.py` (Classe `GraphNegativeSampler`)
  Strutture per codificare linguisticamente i nodi usando SentenceTransformers e corrompere logicamente gli archi.

---

### 3. Modulo: Inferenza ed Analisi Cinetica (Inference)
Applica il modello pre-addestrato a testi nuovi per quantificare, analizzare matematicamente e mappare lo Spazio Latente.

- `Inference/book_embedder.py` (Classe **`BookGraphEmbedder`**)
  Passa un nuovo Knowledge Graph inedito attraverso la GNN pre-allenata. Applica strategie di Pooling differenziate (Mean, Max, Sum, Attention) per definire la *fingerprint strutturale* dell'intera opera.
- `Inference/similarity_analyzer.py` (Classe **`SemanticSimilarityAnalyzer`**)
  Elabora il dataset vettoriale prodotto dall'Embedder eseguendo la *Cosine Similarity* tramite normalizzazione PyTorch (Tutti v Tutti), stampando heatmap ad altissima precisione con pre-processing ast Pydantic integrato.
- `Inference/latent_space_visualization.py` (Classe **`LatentSpaceVisualizer`**)
  Riunisce e ottimizza sia gli algoritmi di riduzione  T-SNE che UMAP in un singolo pattern strategico OOP, generando Scatterplots colorati Plotly in HTML.

---

*Refactoring architetturale OOP curato dinamicamente con successo per massimizzare estendibilità e manutenibilità del codice.*
