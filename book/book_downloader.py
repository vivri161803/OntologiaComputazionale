import os
import time
import requests
from typing import Dict

class BookDownloader:
    """
    Classe responsabile per il download e il salvataggio strutturato dei file testuali 
    dei romanzi (es. da Project Gutenberg Australia).
    """
    def __init__(self, output_dir: str = "book/txt"):
        self.output_dir = output_dir
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_book(self, url: str) -> str:
        """Esegue la HTTP GET request per acquisire il payload testuale grezzo."""
        print(f"  -> Download in corso da {url}...")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        response.encoding = response.apparent_encoding 
        return response.text
        
    def save_book(self, title: str, text: str) -> str:
        """Archivia il testo nel file system sotto la cartella di output e ne restituisce il path."""
        file_path = os.path.join(self.output_dir, f"{title}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return file_path

    def process_library(self, library: Dict[str, str], delay: int = 2):
        """Scarica e salva in loop tutti i libri forniti nel dizionario {Titolo: URL}."""
        print(f"Inizio pipeline di download per {len(library)} libri...\n")
        
        for title, url in library.items():
            print(f"Elaborazione: {title}")
            try:
                raw_text = self.download_book(url)
                saved_path = self.save_book(title, raw_text)
                print(f"  [OK] {title} salvato con successo in {saved_path}!\n")
            except Exception as e:
                print(f"  [ERRORE] Impossibile elaborare {title}: {e}\n")
            
            # Cortesia per non sovraccaricare il server
            time.sleep(delay)

if __name__ == "__main__":
    # Esempio d'uso
    CLASSICI_AUSTRALIA = {
        "The_Time_Machine": "http://gutenberg.net.au/ebooks06/0609221.txt",
        "Dracula": "http://gutenberg.net.au/ebooks/fr100055.txt",
        "Burmese_Days": "http://gutenberg.net.au/ebooks02/0200051.txt",
        "Coming_Up_For_Air": "http://gutenberg.net.au/ebooks02/0200031.txt",
        "Keep_the_Aspidistra_Flying": "http://gutenberg.net.au/ebooks02/0200021.txt",
        "Down_And_Out_In_Paris_And_London": "http://gutenberg.net.au/ebooks03/0300011.txt",
        "A_Room_Of_Ones_Own": "http://gutenberg.net.au/ebooks02/0200791.txt",
        "The_Three_Hostages": "http://gutenberg.net.au/ebooks03/0301231.txt",
        "Last_And_First_Men": "http://gutenberg.net.au/ebooks06/0601271.txt",
        "You_Cant_Go_Home_Again": "http://gutenberg.net.au/ebooks07/0700231.txt"
    }
    
    downloader = BookDownloader(output_dir="book/txt")
    downloader.process_library(CLASSICI_AUSTRALIA)
