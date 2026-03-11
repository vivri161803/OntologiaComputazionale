from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Optional

# --- ENTITÀ A VALORE CHIUSO ---

class Role(str, Enum):
    SOGGETTO_EROE = "SOGGETTO_EROE" # [cite: 46]
    OGGETTO_DI_VALORE = "OGGETTO_DI_VALORE" # [cite: 47]
    DESTINANTE_MANDANTE = "DESTINANTE_MANDANTE" # [cite: 49]
    DESTINATARIO_BENEFICIARIO = "DESTINATARIO_BENEFICIARIO" # [cite: 50]
    OPPONENTE_ANTAGONISTA = "OPPONENTE_ANTAGONISTA" # [cite: 51]
    AIUTANTE_MAGICO = "AIUTANTE_MAGICO" # [cite: 52]
    DONATORE_FORNITORE = "DONATORE_FORNITORE" # [cite: 53]
    FALSO_EROE = "FALSO_EROE" # [cite: 54]

class EventType(str, Enum):
    # Fasi Macro-Strutturali
    ESPOSIZIONE = "ESPOSIZIONE" # [cite: 58]
    PREDICAMENTO = "PREDICAMENTO" # [cite: 58]
    DISTRICAMENTO = "DISTRICAMENTO" # [cite: 59]
    # Azioni di Trama
    PREPARAZIONE = "PREPARAZIONE" # [cite: 62]
    LOTTA_SCONTRO = "LOTTA_SCONTRO" # [cite: 63]
    VITTORIA = "VITTORIA" # [cite: 64]
    SCONFITTA = "SCONFITTA" # [cite: 65]
    RITORNO = "RITORNO" # [cite: 67]
    SACRIFICIO = "SACRIFICIO" # [cite: 68]
    # Snodi Logici
    CAMBIAMENTO_DI_STATO = "CAMBIAMENTO_DI_STATO" # [cite: 71]
    RIVELAZIONE_SCOPERTA = "RIVELAZIONE_SCOPERTA" # [cite: 72]
    ALLEANZA_PATTO = "ALLEANZA_PATTO" # [cite: 73]
    TRADIMENTO_ROTTURA = "TRADIMENTO_ROTTURA" # [cite: 75]
    SPOSTAMENTO_VIAGGIO = "SPOSTAMENTO_VIAGGIO" # [cite: 76]

class LocationType(str, Enum):
    AL_CHIUSO = "AL_CHIUSO" # [cite: 80]
    ALL_APERTO = "ALL_APERTO" # [cite: 81]
    SPAZIO_STRATEGICO = "SPAZIO_STRATEGICO" # [cite: 84]
    SPAZIO_AFFETTIVO = "SPAZIO_AFFETTIVO" # [cite: 85]
    CAMPO_DI_BATTAGLIA = "CAMPO_DI_BATTAGLIA" # [cite: 86]
    LA_STRADA = "LA_STRADA" # [cite: 89]
    IL_SALOTTO = "IL_SALOTTO" # [cite: 89]
    IL_CASTELLO = "IL_CASTELLO" # [cite: 90]
    LA_SOGLIA = "LA_SOGLIA" # [cite: 91]
    RIFUGIO = "RIFUGIO" # [cite: 92]

# --- ARCHI DIREZIONATI (RELAZIONI) ---

class EdgeLabel(str, Enum):
    # Strutturali e di Contesto
    COME = "COME" # [cite: 20]
    PROVIENE_DA = "PROVIENE_DA" # [cite: 22]
    AVVIENE_IN = "AVVIENE_IN" # [cite: 23]
    AVVIENE_DOPO = "AVVIENE_DOPO" # [cite: 25]
    
    # Dinamici (PARTECIPA_COME_...) [cite: 27, 28, 94]
    PARTECIPA_COME_SOGGETTO = "PARTECIPA_COME_SOGGETTO" # [cite: 97]
    PARTECIPA_COME_OGGETTO_BERSAGLIO = "PARTECIPA_COME_OGGETTO_BERSAGLIO" # [cite: 98]
    PARTECIPA_COME_AIUTANTE_ALLEATO = "PARTECIPA_COME_AIUTANTE_ALLEATO" # [cite: 102]
    PARTECIPA_COME_OPPONENTE_RIVALE = "PARTECIPA_COME_OPPONENTE_RIVALE" # [cite: 103]
    PARTECIPA_COME_TRADITORE = "PARTECIPA_COME_TRADITORE" # [cite: 104]
    PARTECIPA_COME_DESTINANTE = "PARTECIPA_COME_DESTINANTE" # [cite: 110]
    PARTECIPA_COME_DESTINATARIO = "PARTECIPA_COME_DESTINATARIO" # [cite: 111]
    PARTECIPA_COME_DONATORE = "PARTECIPA_COME_DONATORE" # [cite: 112]
    PARTECIPA_COME_AMANTE_AMATO = "PARTECIPA_COME_AMANTE_AMATO" # [cite: 116]
    PARTECIPA_COME_CONIUGE_FAMILIARE = "PARTECIPA_COME_CONIUGE_FAMILIARE" # [cite: 118]
    PARTECIPA_COME_AMICO = "PARTECIPA_COME_AMICO" # [cite: 120]
    PARTECIPA_COME_SPETTATORE_NEUTRALE = "PARTECIPA_COME_SPETTATORE_NEUTRALE" # [cite: 121]

# --- 1. CLASSE: ENTITÀ A VALORE APERTO ---
class OpenNode(BaseModel):
    id: str = Field(..., description="Un ID univoco, es. 'char_frodo', 'evt_duello'")
    label: str = Field(..., description="Il testo libero estratto dal libro")
    node_type: str = Field(..., description="Deve essere 'Character', 'Event', o 'Location'")

# --- 2. CLASSE: ENTITÀ A VALORE CHIUSO (La tua intuizione) ---
class ClosedNode(BaseModel):
    # Usiamo Union per dire che il valore deve appartenere a uno dei nostri 3 vocabolari
    id: str = Field(..., description="Un ID univoco, es. 'char_frodo', 'evt_duello'")
    label: Union[Role, EventType, LocationType] = Field(..., description="Il valore universale esatto")
    node_type: str = Field(..., description="Deve essere 'Role', 'EventType', o 'LocationType'")

# --- 3. TRIPLETTE (Archi) ---
class Relation(BaseModel):
    source_id: str = Field(..., description="L'ID del nodo OpenNode di partenza")
    edge_label: EdgeLabel = Field(..., description="L'etichetta direzionale chiusa")
    
    # --- LA MODIFICA È QUI: Sdoppiamo i target per forzare la validazione ---
    target_open_id: Optional[str] = Field(
        default=None, 
        description="L'ID del nodo di arrivo. DA USARE SOLO se il target è un OpenNode (es. Personaggio, Luogo o Evento)."
    )
    target_closed_value: Optional[Union[Role, EventType, LocationType]] = Field(
        default=None, 
        description="Il valore universale esatto. DA USARE OBBLIGATORIAMENTE E SOLO se l'arco è 'COME'."
    )
    
# --- OUTPUT FINALE ---
class SubGraphExtraction(BaseModel):
    open_nodes: List[OpenNode] = Field(..., description="I nodi estratti dal testo (valore aperto)")
    closed_nodes: List[ClosedNode] = Field(..., description="I nodi estratti dal testo (valore chiuso)")
    # Non serve far estrarre i closed_nodes all'LLM, perché sono universali e già noti! 
    # L'LLM deve solo usarli come 'target' nelle relations.
    relations: List[Relation] = Field(..., description="Le triplette che collegano i nodi")

# --- Normalizzazione Etichette
class EntityMapping(BaseModel):
    mapping: Dict[str, str] = Field(
        ..., 
        description="Dizionario di normalizzazione. Chiave: ID originale estratto. Valore: ID canonico unificato. Es: {'char_il_portatore': 'char_frodo', 'char_frodo_baggins': 'char_frodo'}"
    )