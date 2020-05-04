import argparse
from collections import defaultdict
import json
import os
import re
import requests
from tqdm import tqdm

from classification.re_config import RELATIONS_ENTITY_TYPES_FOR_SEARCH
from scripts.search.download_search_examples import download_from_spike_search, get_file_names, merge_and_save_examples

EXPLAIN_URL = 'http://35.246.149.250:5000/'

PLACEHOLDERS = {'PERSON': "John"}
SCRIPT_DIR = 'scripts/search'
LIMIT = -1

def main(args):
    download_dir = os.path.join(SCRIPT_DIR, 'search_from_generation')
    output_dir = os.path.join('data', args.dataset, 'search', 'search_from_generation')
    positive_outfiles, negative_outfiles = None, None
    if args.download_explanations:
        create_explanation_dictionary(args, download_dir)
    explanations = json.load(open(os.path.join(download_dir, args.relation+'_explanations.json'), 'r'))

    patterns = {args.relation: create_patterns_with_triggers(explanations)}
    if args.download_examples:
        positive_outfiles = download_from_spike_search(download_dir, patterns, LIMIT)
    if args.merge_patterns:
        positive_outfiles, _ = get_file_names(download_dir)
        _, negative_outfiles = get_file_names(os.path.join(SCRIPT_DIR, 'negs'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        relations_num_rows = merge_and_save_examples(positive_outfiles, negative_outfiles, output_dir, patterns, args.dataset)

        save_file_lengths(os.path.join(output_dir, 'file_lengths.json'), relations_num_rows)

def save_file_lengths(file_path, relations_num_rows):
    if not os.path.exists(file_path):
        lengths = relations_num_rows
    else:
        lengths = json.load(open(file_path, 'r'))
        for k, v in relations_num_rows.items():
            lengths[k] = v
    
    with open(file_path, 'w') as file:
        json.dump(lengths, file)

def create_explanation_dictionary(args, download_dir):
    with open(args.generation_file, 'r') as f:
        gens = f.readlines()

    explanations = defaultdict(list)
    search_query_api = '/api/3/search/query'
    for gen in tqdm(gens):
        search_query_params = query_params(args.relation, gen.rstrip())

        request = requests.post(url=EXPLAIN_URL + search_query_api,
                                headers={"Content-Type": "application/json"},
                                data=json.dumps(search_query_params))

        assert request.status_code == 204,  search_query_params
        explain_location = request.headers['Explain-Location']

        request = requests.get(url=EXPLAIN_URL + explain_location, headers={"Content-Type": "application/json"})
        explain = json.loads(request.content)['graph']
        explanations[str(explain)].append(search_query_params['queries']['syntactic'])

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    with open(os.path.join(download_dir, args.relation+'_explanations.json'), 'w') as f:
        json.dump(explanations, f)

def create_patterns_with_triggers(explanations):
    patterns = []
    for _, gens in explanations.items():
        if len(gens) == 1: continue
        triggers = set()
        for gen in gens:
            triggers.add(re.findall(r"\[\$ (.*?)\]", gen)[0])
        patterns.append(re.sub(r"\[\$ (.*?)\]", f"[t:w={'|'.join(triggers)} {next(iter(triggers))}]", gens[0]))
    return patterns

def query_params(relation, gen):
    return {
        "queries": {
            "syntactic": convert_wrapped_gen_to_query(relation, gen)
        },
        "data_set_name": "wikipedia"
    }

def switch_ent_to_spike_syntax(gen, ent_id, entity_type):
    E = f"E{ent_id}"
    found_ent = re.findall(rf"\[{E}\] (.*?) \[\/{E}\]", gen)
    assert len(found_ent) == 1
    if found_ent[0] in ["he", "she", "her", "his"]:
        replace_with = f"{{e{ent_id} {found_ent[0]}}}"
    else:
        replace_with = f"{{e{ent_id}:e={entity_type} {PLACEHOLDERS[entity_type]}}}"
    return re.sub(rf"\[{E}\] {found_ent[0]} \[\/{E}\]", replace_with, gen)

def convert_wrapped_gen_to_query(relation, gen):
    entity_type1, entity_type2 = RELATIONS_ENTITY_TYPES_FOR_SEARCH[relation].split(':')
    gen = switch_ent_to_spike_syntax(gen, 1, entity_type1)
    gen = switch_ent_to_spike_syntax(gen, 2, entity_type2)

    return gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=['tacred', 'docred'])
    parser.add_argument("--download_explanations", action='store_true')
    parser.add_argument("--download_examples", action='store_true')
    parser.add_argument("--merge_patterns", action='store_true')
    args = parser.parse_args()
    main(args)