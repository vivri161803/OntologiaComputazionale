from flask import Flask, render_template, request
import sys
import os

# Assicuriamoci che il modulo root (OntologiaComputazionale) sia nel path per i sub-moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metadata_fetcher import BookMetadataFetcher
from similarity_service import SimilarityEngine
from train_set import train_library, test_library

app = Flask(__name__)

# Istanziamo i servizi all'avvio: Singletons (evitano di ricaricare i pesi GNN)
print("[INFO] Loading Model & API Services...")
fetcher = BookMetadataFetcher()
engine = SimilarityEngine()
print("[INFO] App Ready!")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Estraiamo la lista completa dei TSV e creiamo opzioni leggibili
    all_books = train_library + test_library
    available_books = [{"path": p, "title": fetcher._clean_title(p)} for p in all_books]
    
    if request.method == 'POST':
        book_path = request.form.get('book_path')
        if book_path:
            # Calcoliamo i Top K dal Modello
            top_k_results = engine.get_top_k(book_path, k=3)
            
            # Fetch Metadata per Target (OpenLibrary)
            target_metadata = fetcher.fetch_metadata(book_path)
            
            # Fetch Metadata per Raccomandazioni (OpenLibrary)
            recommendations = []
            for res in top_k_results:
                meta = fetcher.fetch_metadata(res["path"])
                meta["score"] = res["score"] # Inseriamo l'affinità
                recommendations.append(meta)
                
            return render_template('index.html', 
                                   books=available_books, 
                                   target=target_metadata, 
                                   recommendations=recommendations, 
                                   selected=book_path)
                                   
    # Se GET, mostra solo la landing page pulita
    return render_template('index.html', books=available_books)

if __name__ == '__main__':
    # Esegue l'app sulla porta 5555 in locale.
    app.run(host='127.0.0.1', port=5555, debug=True)
