import os
import requests
import time

# ==========================================
# 1. LA BIBLIOTECA (Gutenberg Australia)
# Dizionario: { "Nome_Libro": "URL_Diretto_al_TXT" }
# ==========================================
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

# Cartella di destinazione
OUTPUT_DIR = "./txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def scarica_libro(url):
    """Scarica il testo grezzo direttamente dall'URL fornito."""
    print(f"  -> Download in corso da {url}...")
    # Aggiungiamo un header (User-Agent) perché alcuni siti bloccano gli script Python anonimi
    headers = {'User-Agent': 'Mozilla/5.0'} 
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    # Proviamo a decodificare in utf-8, se fallisce usiamo latin-1 (comune sui siti vecchi)
    response.encoding = response.apparent_encoding 
    return response.text

# ==========================================
# 2. PIPELINE PRINCIPALE
# ==========================================
print(f"Inizio pipeline di download (Australia) per {len(CLASSICI_AUSTRALIA)} libri...\n")

for titolo, url in CLASSICI_AUSTRALIA.items():
    print(f"Elaborazione: {titolo}")
    
    try:
        # Scarica il testo intero
        testo_grezzo = scarica_libro(url)
        
        # Genera il percorso del file (es: txt/Dracula.txt)
        percorso_file = os.path.join(OUTPUT_DIR, f"{titolo}.txt")
        
        # Salva l'intero testo in un unico file
        with open(percorso_file, 'w', encoding='utf-8') as f:
            f.write(testo_grezzo)
            
        print(f"  [OK] {titolo} salvato con successo in {percorso_file}!\n")
        
    except Exception as e:
        print(f"  [ERRORE] Impossibile elaborare {titolo}: {e}\n")
    
    # Pausa di cortesia per non sovraccaricare i server
    time.sleep(2)