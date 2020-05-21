import argparse
from collections import defaultdict
import csv
import json
import os
import re
import requests
from tqdm import tqdm

from classification.re_config import RELATIONS_ENTITY_TYPES_FOR_SEARCH
from scripts.search.download_search_examples import (download_from_spike_search,
                                                     get_file_names,
                                                     merge_and_save_examples,
                                                     map_array_given_header,
                                                     update_file_lengths)

EXPLAIN_URL = 'http://34.89.233.227:5000'

PLACEHOLDERS = {'PERSON': "John", "ORGANIZATION": "Microsoft", "LOCATION": "Israel", "MISC": "American", "DATE": "November", "ORGANIZATION|MISC": "Microsoft", "PERSON|ORGANIZATION|LOCATION": "John"}
SCRIPT_DIR = 'scripts/search'
LIMIT = -1

def main(args):
    download_dir = os.path.join(SCRIPT_DIR, 'search_from_generation', args.dataset, args.relation)
    capped_dataset_name = 'DocRED' if args.dataset == 'docred' else 'tacred'
    output_dir = os.path.join('data', capped_dataset_name, 'search', 'search_from_generation')
    positive_outfiles, negative_outfiles = None, None
    if args.download_explanations:
        create_explanation_dictionary(args, download_dir)
    explanations = json.load(open(os.path.join(download_dir, args.relation+'_explanations.json'), 'r'))

    explanations_with_trigger_patterns = merge_patterns_with_triggers(args.relation, explanations)
    patterns = {args.relation: [p for p in explanations_with_trigger_patterns.values()]}
    if args.download_examples:
        positive_outfiles = download_from_spike_search(download_dir, patterns, LIMIT)

    positive_outfiles, _ = get_file_names(download_dir)
    _, negative_outfiles = get_file_names(os.path.join(SCRIPT_DIR, 'negs'))
    if args.merge_patterns:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        relations_num_rows = merge_and_save_examples({args.relation: positive_outfiles[args.relation]}, negative_outfiles, output_dir, patterns, args.dataset)

        update_file_lengths(os.path.join(output_dir, 'file_lengths.json'), relations_num_rows)

    if args.evaluate:
        print_downloaded_examples(args.relation, positive_outfiles, explanations_with_trigger_patterns, {k: len(v) for k, v in explanations.items()})

def print_downloaded_examples(relation, positive_files, explanations, explanations_counts):
    relation_files = positive_files[relation]
    assert len(relation_files) == len(explanations)
    relation_files.sort(key = lambda f : int(f.split('-')[-1]))
    print(f"total of {len(relation_files)} patterns")
    for file, explain, explain_example in zip(relation_files, explanations.keys(), explanations.values()):
        reader = csv.reader(open(file, 'r'), delimiter='\t')
        examples = [l for l in reader]
        print(f"\x1b[32mFile: {file}. Number downloaded: {len(examples) - 1} \x1b[00m")
        print(f"\x1b[32mNumber annotated in gen file: {explanations_counts[explain]}\x1b[00m")
        print(explain)
        print(explain_example)
        for d in examples[1:6]:
            d = map_array_given_header(d, examples[0])
            colored_sent = d['sentence_text'].replace(d['e1'], f"\x1b[32m{d['e1']}\x1b[00m").replace(d['e2'], f"\x1b[35m{d['e2']}\x1b[00m")
            print(f"(\x1b[32m{d['e1']}\x1b[00m, \x1b[35m{d['e2']}\x1b[00m) >>> {colored_sent}")
        print()

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
        explain['edges'].sort(key=lambda d: (d['source'], d['target'], d['label']))
        if not pattern_to_add(args.dataset, args.relation, explain): continue
        explanations[str(explain)].append(search_query_params['queries']['syntactic'])

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    with open(os.path.join(download_dir, args.relation+'_explanations.json'), 'w') as f:
        json.dump(explanations, f)

def pattern_to_add(dataset, relation, explain):
    # need to move this outside
    def s(l):
        return sorted(l, key=lambda d: (d['source'], d['target'], d['label']))
    if relation == 'per:children':
        if explain == {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubjpass'},
                                                 {'source': 1, 'target': 2, 'label': 'nmod'},
                                                 {'source': 1, 'target': 3, 'label': 'nmod'},
                                                 {'source': 2, 'target': 3, 'label': 'conj'}
                                               ])}:
            return False
    if relation == 'per:city_of_death':
        if explain in [
            {'roots': [0], 'edges': s([{'source': 0, 'target': 2, 'label': 'parataxis'},
                                       {'source': 2, 'target': 1, 'label': 'nsubj'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'compound'},
                                       {'source': 2, 'target': 1, 'label': 'nsubj'},
                                       {'source': 2, 'target': 4, 'label': 'ccomp'},
                                       {'source': 4, 'target': 3, 'label': 'nsubj'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'compound'},
                                       {'source': 2, 'target': 1, 'label': 'nsubj'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj'},
                                       {'source': 1, 'target': 2, 'label': 'dobj'},
                                       {'source': 2, 'target': 4, 'label': 'nmod'},
                                       {'source': 4, 'target': 3, 'label': 'compound'}])},
            {'roots': [0], 'edges': s([{'source': 0, 'target': 1, 'label': 'nmod'},
                                       {'source': 2, 'target': 0, 'label': 'nsubj'},
                                       {'source': 2, 'target': 3, 'label': 'nmod'}])}
        ]:
            return False
    if relation == 'per:spouse':
        if explain in [
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj'},
                                       {'source': 1, 'target': 3, 'label': 'nmod'},
                                       {'source': 3, 'target': 2, 'label': 'compound'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj'},
                                       {'source': 1, 'target': 2, 'label': 'nmod'},
                                       {'source': 2, 'target': 3, 'label': 'nmod'},
                                       {'source': 3, 'target': 4, 'label': 'appos'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nmod'},
                                       {'source': 1, 'target': 2, 'label': 'appos'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj'},
                                       {'source': 1, 'target': 3, 'label': 'dobj'},
                                       {'source': 3, 'target': 2, 'label': 'nmod'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj'},
                                       {'source': 1, 'target': 3, 'label': 'conj'},
                                       {'source': 3, 'target': 2, 'label': 'nsubjpass'}])}

        ]:
            return False
    if relation == 'per:religion':
        if explain in [
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj:xsubj'},
                                       {'source': 1, 'target': 3, 'label': 'dobj'},
                                       {'source': 3, 'target': 2, 'label': 'amod'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubj'},
                                       {'source': 1, 'target': 3, 'label': 'nmod'},
                                       {'source': 3, 'target': 2, 'label': 'amod'}])},
            {'roots': [0], 'edges': s([{'source': 2, 'target': 0, 'label': 'ccomp'},
                                       {'source': 2, 'target': 1, 'label': 'nsubj'}])},
            {'roots': [0], 'edges': s([{'source': 0, 'target': 1, 'label': 'nmod'}])}
        ]:
            return False
    if relation == 'per:religion':
        if explain in [
            {'roots': [0], 'edges': s([{'source': 0, 'target': 2, 'label': 'conj'},
                                       {'source': 0, 'target': 3, 'label': 'appos'},
                                       {'source': 2, 'target': 1, 'label': 'amod'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'compound'}])},
            {'roots': [0], 'edges': [{'source': 0, 'target': 1, 'label': 'appos'}]}
            ]:
            return False
    if relation == 'org:country_of_headquarters':
        if explain in [
            {'roots': [0], 'edges': s([{'source': 0, 'target': 1, 'label': 'dobj'},
                                       {'source': 0, 'target': 5, 'label': 'dep'},
                                       {'source': 1, 'target': 2, 'label': 'appos'},
                                       {'source': 2, 'target': 3, 'label': 'nmod'},
                                       {'source': 3, 'target': 4, 'label': 'acl'}])},
            {'roots': [0], 'edges': s([{'source': 0, 'target': 2, 'label': 'nmod'},
                                       {'source': 2, 'target': 1, 'label': 'case'}])}
        ]:
            return False

    if relation == 'org:country_of_headquarters':
        if explain in [
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubjpass'},
                                       {'source': 1, 'target': 3, 'label': 'nmod'},
                                       {'source': 3, 'target': 2, 'label': 'case'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'nsubjpass'},
                                       {'source': 1, 'target': 3, 'label': 'advcl_in'},
                                       {'source': 3, 'target': 2, 'label': 'case'}])},
            {'roots': [0], 'edges': s([{'source': 0, 'target': 1, 'label': 'dep'}])},
            {'roots': [0], 'edges': s([{'source': 0, 'target': 1, 'label': 'case'},
                                       {'source': 2, 'target': 0, 'label': 'nmod'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'amod'},
                                       {'source': 2, 'target': 1, 'label': 'nsubj'},
                                       {'source': 2, 'target': 3, 'label': 'advcl_in'},
                                       {'source': 3, 'target': 4, 'label': 'dobj'},
                                       {'source': 4, 'target': 7, 'label': 'nmod'},
                                       {'source': 7, 'target': 5, 'label': 'case'},
                                       {'source': 7, 'target': 6, 'label': 'nmod'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'compound'},
                                       {'source': 1, 'target': 2, 'label': 'appos'}])},
            {'roots': [0], 'edges': s([{'source': 1, 'target': 0, 'label': 'amod'},
                                       {'source': 1, 'target': 3, 'label': 'acl:relcl'},
                                       {'source': 3, 'target': 2, 'label': 'nsubj'}])},
        ]:
            return False

    return True

def enough_similar_patterns(relation, gens):
    if len(gens) == 1:
        return False
    if relation == 'per:spouse' and len(gens) == 2:
        return False
    return True

def merge_patterns_with_triggers(relation, explanations):
    patterns = {}
    for explain, gens in explanations.items():
        if not enough_similar_patterns(relation, gens): continue
        triggers = set()
        for gen in gens:
            trigger = re.findall(r"\[\$ (.*?)\]", gen)
            if trigger:
                assert len(trigger) == 1, f"gen: {gen}\ntrigger:{trigger}"
                trigger = trigger[0]
                if len(trigger.split(' ')) != 1:
                    trigger = ''.join(trigger.split(' '))
                    print(f"This trigger might look weird: {trigger}")
                triggers.add(trigger)

        if triggers:
            patterns[explain] = re.sub(r"\[\$ (.*?)\]", f"[t:w={'|'.join(triggers)} {trigger}]", gens[-1])
        else:
            patterns[explain] = gens[-1]
    return patterns

def query_params(relation, gen):
    return {
        "queries": {
            "syntactic": convert_wrapped_gen_to_query(relation, gen)
        },
        "data_set_name": "wikipedia"
    }

def switch_ent_to_spike_syntax(gen, ent_id, entity_type, relation):
    E = f"E{ent_id}"
    found_ent = re.findall(rf"\[{E}\] (.*?) \[\/{E}\]", gen)
    assert len(found_ent) == 1, f"found weird entity: {found_ent} in gen: {gen}"
    if found_ent[0] in ["her", "his"]:
        replace_with = f"{{e{ent_id}:e {found_ent[0]}}}"
    elif relation == 'per:religion' and entity_type == 'MISC':
        religion_triggers = "Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Islamist|Islam|Muslim|Shia"
        replace_with = f"[e{ent_id}:w={religion_triggers} Jewish]"
    elif relation == 'per:religion' and entity_type == 'MISC':
        replace_with = f"[e{ent_id}:e=MISC {PLACEHOLDERS[entity_type]}]"
    else:
        replace_with = f"{{e{ent_id}:e={entity_type} {PLACEHOLDERS[entity_type]}}}"
    gen = re.sub(rf"\[{E}\] {re.escape(found_ent[0])} \[\/{E}\]", replace_with, gen)
    gen = re.sub(r"\[t ' s]", r"[pos:t=POS ']", gen)
    gen = re.sub(r"\[t 's]", r"[pos:t=POS ']", gen)
    return re.sub(r"\[t", r"[$", gen)

def convert_wrapped_gen_to_query(relation, gen):
    entity_type1, entity_type2 = RELATIONS_ENTITY_TYPES_FOR_SEARCH[relation].split(':')

    gen = switch_ent_to_spike_syntax(gen, 1, entity_type1, relation)
    gen = switch_ent_to_spike_syntax(gen, 2, entity_type2, relation)

    return gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=['tacred', 'docred'])
    parser.add_argument("--download_explanations", action='store_true')
    parser.add_argument("--download_examples", action='store_true')
    parser.add_argument("--merge_patterns", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    args = parser.parse_args()
    main(args)