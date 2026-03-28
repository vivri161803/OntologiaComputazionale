# Ontologia Computazionale: GNN-based Narrative Analysis & Recommendation

Questo progetto implementa una pipeline avanzata di **Analisi Narratologica** che trasforma testi letterari in **Knowledge Graphs** (KG) strutturati utilizzando LLM, per poi analizzarli attraverso **Graph Neural Networks** (GNN). Il sistema è in grado di mappare lo spazio latente delle opere letterarie e consigliare romanzi basandosi sulla loro affinità strutturale e ontologica.

---

## 🚀 Cosa è stato fatto

Il progetto copre l'intero ciclo di vita del dato, dall'estrazione grezza alla visualizzazione interattiva:

1.  **Estrazione KG con LLM**: Pipeline che utilizza **Claude Haiku** (via `instructor`) per estrarre entità e relazioni dai testi dei libri, applicando logiche di **Entity Resolution** per unificare i personaggi (es. "Aragorn" e "Grampasso").
2.  **Modellazione GNN**: Architettura basata su **R-GCN** (Relational Graph Convolutional Networks) e decoder **TransE** per l'apprendimento di graph embeddings. Include negative sampling e ottimizzazione degli iperparametri con **Optuna**.
3.  **Analisi dello Spazio Latente**: Utilizzo di **UMAP** e **T-SNE** per proiettare i complessi embedding dei grafi in uno spazio 2D, permettendo di visualizzare visivamente le "distanze" tra le opere.
4.  **Motore di Similarità**: Sistema di calcolo della *Cosine Similarity* tra i vettori strutturali delle opere per identificare i "match" narratologici più stretti.
5.  **Web Showcase**: Applicazione interattiva basata su **Flask** con interfaccia **Glassmorphism**, che integra metadati e cover scaricate dinamicamente via **OpenLibrary API**.

---

## 🛠️ Stack Tecnologico

Il progetto sfrutta le tecnologie più moderne nel campo dell'AI e del Graph ML:

*   **Linguaggio**: Python 3.12+
*   **Gestione Ambiente**: `uv` (per performance fulminee e determinismo delle dipendenze).
*   **LLM & Orchestrazione**: Anthropic Claude Haiku, `instructor`, `Pydantic`.
*   **Graph Machine Learning**: `PyTorch`, `PyTorch Geometric` (R-GCN, TransE).
*   **Elaborazione Grafi**: `NetworkX`, `PyVis`.
*   **Natural Language Processing**: `Sentence-Transformers` (SBERT) per le feature iniziali dei nodi.
*   **Data Science**: `Pandas`, `NumPy`, `Scikit-Learn`, `UMAP-learn`.
*   **Visualizzazione**: `Plotly`, `Matplotlib`, `Seaborn`.
*   **Web Stack**: `Flask`, `Requests` (OpenLibrary API integration).

---

## 📂 Struttura del Progetto

*   **`KG_Extraction/`**: Core dell'acquisizione. Contiene l'estrattore LLM, il merger di grafi e l'esportatore TSV.
*   **`GNN/`**: Definizione del modello neurale, script di training, negative sampling e ottimizzazione.
*   **`Inference/`**: Moduli per generare embedding di nuovi libri e analizzare le similarità.
*   **`app/`**: Il backend e frontend della Web Application demo.
*   **`book/`**: Contiene i testi originali (`.txt`), i grafi estratti (`.tsv`) e i database locali.
*   **`TrainedModel/`**: Pesi del modello allenato e mappature delle relazioni.

---

## 🏁 Come Iniziare

Assicurati di avere `uv` installato sul tuo sistema.

### 1. Eseguire la Demo End-to-End
Per estrarre un grafo, generare i grafici dello spazio latente e calcolare le similarità in un colpo solo:
```bash
uv run demo.py
```

### 2. Avviare la Web App
Per navigare tra i consigli dei libri con un'interfaccia premium:
```bash
uv run python app/app.py
```
L'app sarà disponibile su **[http://127.0.0.1:5555/](http://127.0.0.1:5555/)**.

---

*Progetto sviluppato come framework modulare in Object-Oriented Programming (OOP) per la ricerca in Computer Science ed Humanities.*
