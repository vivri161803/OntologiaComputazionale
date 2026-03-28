import requests
import urllib.parse
import os

class BookMetadataFetcher:
    """
    Classe OOP responsabile di fetchare i metadati di un libro
    appoggiandosi all'API pubblica di Open Library.
    """
    def __init__(self):
        self.search_url = "https://openlibrary.org/search.json"
        
    def _clean_title(self, path: str) -> str:
        """
        Pulisce il nome file del TSV per ricavare un titolo passabile all'API.
        """
        filename = os.path.basename(path)
        # Sostituiamo underscore con spazio e rimuoviamo sub_grafi, _kge, .tsv
        title = filename.replace('_', ' ').replace('.tsv', '')
        title = title.replace('sub grafi', '').replace(' kge', '')
        # Special cases per fixare titoli attaccati (es. JekyllHyde)
        if title.lower().startswith('jekyll'):
            title = 'Dr. Jekyll and Mr. Hyde'
        elif title.lower().startswith('mobydick'):
            title = 'Moby Dick'
        elif title.lower().startswith('theverdict'):
            title = 'The Verdict'
        elif title.lower().startswith('tenderisthenight'):
            title = 'Tender is the Night'
        elif title.lower().startswith('thegreatgatsby'):
            title = 'The Great Gatsby'
        elif title.lower().startswith('madamebovary'):
            title = 'Madame Bovary'
        
        return title.strip()

    def fetch_metadata(self, book_path: str) -> dict:
        """
        Esegue la chiamata all'API e ritorna un dizionario con Cover URL, Autore e Anno.
        """
        title = self._clean_title(book_path)
        
        params = {
            'title': title,
            'limit': 1
        }
        
        default_cover = "https://via.placeholder.com/300x450/111111/555555?text=Cover+Not+Found"
        result = {
            'title': title,
            'author': 'Unknown Author',
            'year': 'Unknown',
            'cover_url': default_cover
        }
        
        try:
            response = requests.get(self.search_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                docs = data.get('docs', [])
                if docs:
                    book_data = docs[0]
                    # Autore
                    if 'author_name' in book_data and book_data['author_name']:
                        result['author'] = book_data['author_name'][0]
                    # Anno
                    if 'first_publish_year' in book_data:
                        result['year'] = str(book_data['first_publish_year'])
                    # Copertina tramite Cover iD
                    if 'cover_i' in book_data:
                        cover_id = book_data['cover_i']
                        result['cover_url'] = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
                    elif 'isbn' in book_data and book_data['isbn']:
                        # Alternativa fallback su ISBN
                        isbn = book_data['isbn'][0]
                        result['cover_url'] = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
        except Exception as e:
            print(f"[Warning] Errore nel fetching metadati iterando sull'API per '{title}': {e}")
            
        return result
