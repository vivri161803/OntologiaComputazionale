import random
import pandas as pd
from typing import List

class GraphNegativeSampler:
    def __init__(self):
        self.true_triplets = set()
        self.relations = set()
        self.nodes = list()
        
        # Dizionari per mantenere il Negative Sampling "Type-Aware" implicitamente
        self.valid_heads_for_rel = {}
        self.valid_tails_for_rel = {}
        
    def load_tsv(self, tsv_files: List[str]):
        """Carica le triplette direttamente dai TSV e deduce i domini delle relazioni."""
        all_triplets = []
        
        for file in tsv_files:
            df = pd.read_csv(file, sep='\t', names=['head', 'relation', 'tail'], dtype=str)
            
            for _, row in df.iterrows():
                h, r, t = str(row['head']), str(row['relation']), str(row['tail'])
                all_triplets.append((h, r, t))
                self.true_triplets.add((h, r, t))
                self.relations.add(r)
                
                # Popoliamo i domini validi per corruzioni logiche
                if r not in self.valid_heads_for_rel:
                    self.valid_heads_for_rel[r] = set()
                    self.valid_tails_for_rel[r] = set()
                
                self.valid_heads_for_rel[r].add(h)
                self.valid_tails_for_rel[r].add(t)

        unique_nodes = set([t[0] for t in all_triplets] + [t[2] for t in all_triplets])
        self.nodes = list(unique_nodes)

    def generate_dataset(self, k_negatives: int = 3) -> pd.DataFrame:
        dataset = []
        
        for h, r, t in self.true_triplets:
            # Tripletta vera
            dataset.append({"head": h, "relation": r, "tail": t, "label": 1})
            
            for _ in range(k_negatives):
                corrupt_target = random.choices(['head', 'tail', 'relation'], weights=[0.4, 0.4, 0.2])[0]
                h_false, r_false, t_false = h, r, t
                
                if corrupt_target == 'head':
                    # Peschiamo tra i nodi sensati per questo tipo di arco
                    h_false = random.choice(list(self.valid_heads_for_rel.get(r, self.nodes)))
                    
                elif corrupt_target == 'tail':
                    t_false = random.choice(list(self.valid_tails_for_rel.get(r, self.nodes)))
                    
                elif corrupt_target == 'relation' and len(self.relations) > 1:
                    available_rels = list(self.relations - {r})
                    r_false = random.choice(available_rels)
                
                # Controllo anti-collisione
                if (h_false, r_false, t_false) not in self.true_triplets:
                    dataset.append({"head": h_false, "relation": r_false, "tail": t_false, "label": 0})
                    
        df = pd.DataFrame(dataset)
        return df.sample(frac=1).reset_index(drop=True)