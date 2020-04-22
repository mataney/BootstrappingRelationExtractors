import argparse
from collections import Counter
import csv
import os
from tqdm import tqdm

def main(args):
    entities = Counter()

    if args.relation == 'city':
        countries_and_states = read_entities_list(True, True)

    with open(args.file_with_entities, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        e_index = headers.index(args.entity_id)
        for x in tqdm(reader):
            entity = x[e_index]
            if args.relation == 'city':
                if entity in countries_and_states:
                    continue
            entities[entity] += 1

    with open(f'generation_outputs/types/{args.relation}.txt', 'w') as f:
        # for e in entities.most_common(100):
        for e in entities:
            f.write(f"{e}\n")


def read_entities_list(countries, states):
    COUNTRIES_AND_STATES_LOCATION = 'scripts/search/ner_lists'
    ret = set()
    if countries:
        with open(os.path.join(COUNTRIES_AND_STATES_LOCATION, 'countries'), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for x in reader:
                ret.add(x[1])

    if states:
        with open(os.path.join(COUNTRIES_AND_STATES_LOCATION, 'statesandprovinces'), 'r') as f:
            states = f.readlines()
            for s in states:
                ret.add(s.rstrip())

    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_with_entities", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    parser.add_argument("--entity_id", type=str, required=True, choices=['e1', 'e2'])
    args = parser.parse_args()
    main(args)