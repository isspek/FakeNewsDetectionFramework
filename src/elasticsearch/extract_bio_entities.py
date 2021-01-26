from pathlib import Path

import pandas as pd
import spacy
from scispacy.linking import EntityLinker
from tqdm import tqdm


class ScientificEntityLinker:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_sm")
        self.linker = EntityLinker(resolve_abbreviations=True, name="umls")
        self.nlp.add_pipe(self.linker)

    def extract_ents(self, text):
        '''
        Extract biomedical terms/entities from text
        '''
        doc = self.nlp(text)

        # Let's look at a random entity!
        entity = doc.ents[1]

        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
        for umls_ents in entity._.kb_ents:
            for ent in


    def _extract_aliases(self, entity):
        kb_entity = self.linker.kb.cui_to_entity[entity]
        print(kb_entity.canonical_name)
        print(kb_entity.aliases)



if __name__ == '__main__':
    claims_dir = Path('data')
    fakehealth = pd.read_csv(claims_dir / 'FakeHealth.tsv', sep='\t')[:1]
    fakehealth = fakehealth[fakehealth['label'] == 'fake']
    fakehealth = fakehealth.fillna('')  # replace Nan fields
    sel = ScientificEntityLinker()

    for i, row in tqdm(fakehealth.iterrows(), total=len(fakehealth)):
        print(f"Processing document {i}")
        text = row["content"]
        sel.extract_ents(text)
