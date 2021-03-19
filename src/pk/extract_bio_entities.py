from pathlib import Path

import pandas as pd
import spacy
import argparse
from scispacy.linking import EntityLinker
from tqdm import tqdm
import json
import csv


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

        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
        bio_entities = []
        for entity in doc.ents:
            for umls_ents in entity._.kb_ents:
                for ent in umls_ents:
                    try:
                        aliases = self._extract_aliases(ent)
                    except:
                        continue
                    bio_entities += aliases

        bio_entities = [ent.lower() for ent in bio_entities]
        bio_entities = list(set(bio_entities))
        return bio_entities

    def _extract_aliases(self, entity):
        kb_entity = self.linker.kb.cui_to_entity[entity]
        return kb_entity.aliases


def read_fakehealth(path):
    fakehealth = pd.read_csv(path, sep='\t')
    fakehealth = fakehealth[fakehealth['label'] == 'fake']
    fakehealth = fakehealth.fillna('')  # replace Nan fields
    fakehealth = fakehealth['content'].tolist()
    return fakehealth


def read_constraint(path):
    data = pd.read_csv(path, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    data = data['tweet'].tolist()
    return data


def read_coaid(path):
    data = pd.read_csv(path, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep='\t')
    data = data['content'].tolist()
    return data


def read_recovery(path):
    data = pd.read_csv(path, sep='\t')
    data = data['content'].tolist()
    return data


DATASETS = {
    'fakehealth': read_fakehealth,
    'constraint': read_constraint,
    'coaid': read_coaid,
    'recovery': read_recovery

}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    path = Path(args.input_file)
    data = DATASETS[args.data](path)

    sel = ScientificEntityLinker()

    entities_per_doc = []
    for text in tqdm(data, total=len(data)):
        entities = sel.extract_ents(text)
        entities_per_doc.append(entities)

    bio_entities = {'bio_entities': entities_per_doc}
    output_dir = Path('data')
    json.dump(bio_entities, open(args.output_file, 'w'))
