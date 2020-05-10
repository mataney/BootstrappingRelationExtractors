import argparse
from collections import defaultdict
import csv
import json
from itertools import chain
import os
import requests
from tqdm import tqdm
import wget

from classification.re_processors import wrap_text
from classification.re_config import RELATIONS_ENTITY_TYPES_FOR_SEARCH
from scripts.search.download_patterns_config import SINGLE_TRIGGER_PATTERNS, ALL_TRIGGERS_PATTERNS, NEGATIVE_PATTERNS

LIMIT = -1
URL = 'http://34.89.233.227:5000'
SCRIPT_DIR = 'scripts/search'

def main(args):
    capped_dataset_name = 'DocRED' if args.dataset == 'docred' else 'tacred'
    if args.triggers == 'single':
        patterns = SINGLE_TRIGGER_PATTERNS[args.dataset]
        download_dir = os.path.join(SCRIPT_DIR, 'single_trigger_search_results_xxx')
        output_dir = os.path.join('data', capped_dataset_name, 'search', 'single_trigger_search_xxx')
    else:
        patterns = ALL_TRIGGERS_PATTERNS
        download_dir = os.path.join(SCRIPT_DIR, 'all_triggers_search_results_xxx')
        output_dir = os.path.join('data', capped_dataset_name, 'search', 'all_triggers_search_xxx')
    positive_outfiles, negative_outfiles = None, None
    if args.download:
        positive_outfiles = download_from_spike_search(download_dir, patterns, LIMIT)
        # negative_outfiles = download_from_spike_search(download_dir, NEGATIVE_PATTERNS, LIMIT, use_odinson=True)
    if args.merge_patterns:
        if positive_outfiles is None:
            positive_outfiles, _ = get_file_names(download_dir)
        if negative_outfiles is None:
            _, negative_outfiles = get_file_names(os.path.join(SCRIPT_DIR, 'negs'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        relations_num_rows = merge_and_save_examples(positive_outfiles, negative_outfiles, output_dir, patterns, args.dataset)

        update_file_lengths(os.path.join(output_dir, 'file_lengths.json'), relations_num_rows)

def get_file_names(download_dir):
    def get_relation_name_from_file_name(file_name):
        hyps_pos = [i for i, c in enumerate(file_name) if c == '-']
        return file_name[hyps_pos[0]+1:hyps_pos[1]]

    poss, negs = defaultdict(list), defaultdict(list)
    for file in os.listdir(download_dir):
        if 'raw' not in file:
            continue
        relation_name = get_relation_name_from_file_name(file)
        if file.startswith("raw-per") or file.startswith("raw-org"):
            poss[relation_name].append(os.path.join(download_dir, file))
        elif file.startswith("raw-PERSON") or file.startswith("raw-ORGANIZATION"):
            negs[relation_name].append(os.path.join(download_dir, file))
    return poss, negs


def remove_same_sent_id(data):
    grouped = defaultdict(list)
    for d in data:
        grouped[d['sentence_id']].append(d)

    ret = []
    for _, v in grouped.items():
        positive = [d for d in v if d['label'] != 'NOTA']
        assert len(positive) <= 1
        if len(positive) > 0:
            ret.append(positive[0])
        else:
            ret.append(v[-1])
    return ret

def seperate_entities(data):
    if data['e1_first_index'] <= data['e2_first_index']:
        first, second = 'e1', 'e2'
    else:
        first, second = 'e2', 'e1'

    if data[f'{first}_first_index'] < data[f'{second}_first_index'] and \
       data[f'{first}_last_index'] < data[f'{second}_last_index'] and \
       data[f'{first}_last_index'] < data[f'{second}_first_index']:
        return True
    else:
        return False

def entities_validator_for_relation(relation, dataset):
    countries = read_entities_list(countries=True, states=False)
    countries_and_states = read_entities_list(countries=True, states=False)

    if dataset == 'tacred' and relation == "org:country_of_headquarters":
        pass
        # def country_checker(location):
            # return location in countries

        # return country_checker
    elif dataset == 'tacred' and relation == "per:city_of_death":
        pass
        # def city_checker(location):
        #     return not location in countries_and_states

        # return city_checker

    elif relation == "per:origin":
        def non_nationality(nationality):
            return not nationality.lower() in ["republican", "democrat", "rabbi"]

    return lambda ent: True

def read_entities_list(countries, states):
    ret = set()
    if countries:
        with open(os.path.join(SCRIPT_DIR, 'ner_lists', 'countries'), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for x in reader:
                ret.add(x[1])

    if states:
        with open(os.path.join(SCRIPT_DIR, 'ner_lists', 'statesandprovinces'), 'r') as f:
            states = f.readlines()
            for s in states:
                ret.add(s.rstrip())

    return ret

def merge_and_save_examples(positive_outfiles, negative_outfiles, output_dir, patterns, dataset):
    relations_num_rows = {}
    for relation, relation_paths in tqdm(positive_outfiles.items()):
        sent_ids_used_by_relation = merge_positive_examples_and_save(output_dir,
                                                                     relation,
                                                                     relation_paths,
                                                                     patterns[relation],
                                                                     entities_validator_for_relation(relation, dataset),
                                                                     dataset)
        relations_num_rows[relation] = {k: len(v) for k, v in sent_ids_used_by_relation.items()}
        entities = RELATIONS_ENTITY_TYPES_FOR_SEARCH[relation]
        neg_count = merge_negative_examples_and_save_given_relation(output_dir,
                                                                    entities,
                                                                    negative_outfiles[entities],
                                                                    relation,
                                                                    sent_ids_used_by_relation,
                                                                    dataset)
        relations_num_rows[f"{relation}-{entities}"] = neg_count

    return relations_num_rows

def merge_positive_examples_and_save(output_dir, relation, relation_paths, patterns, validate_entities, dataset):
    def used_before(sent_ids_used, sent_id):
        for used in sent_ids_used.values():
            if sent_id in used:
                return True

        return False

    out_file = open(os.path.join(output_dir, relation), 'w')
    writer = csv.writer(out_file, delimiter='\t')
    sent_ids_used = {i: set() for i in range(len(relation_paths))}
    relation_paths.sort(key = lambda f : int(f.split('-')[-1]))
    for i, relation_path in enumerate(relation_paths):
        search_file = open(relation_path, "r", encoding="utf-8")
        print(f"Working on {relation_path}")
        reader = csv.reader(search_file, delimiter='\t')
        headers = next(reader)
        for d in reader:
            d = map_array_given_header(d, headers)
            if not seperate_entities(d) or \
               not validate_entities(d['e2']) or \
               used_before(sent_ids_used, d['sentence_id']):
                continue

            text = wrap_text(d['sentence_text'].split(),
                                d['e1_first_index'],
                                d['e1_last_index'] + 1,
                                d['e2_first_index'],
                                d['e2_last_index'] + 1)
            if dataset == 'docred':
                text = clean_special_tokens(text)

            writer.writerow([text, relation, patterns[i], d['sentence_id']])
            sent_ids_used[i].add(d['sentence_id'])
        search_file.close()
    out_file.close()

    return sent_ids_used

def merge_negative_examples_and_save_given_relation(output_dir, entities, file_paths, relation, positive_ids_used_by_relation, dataset):
    positive_sent_ids_used = set(chain(*positive_ids_used_by_relation.values()))
    last_sent_id_used = -1
    out_file = open(os.path.join(output_dir, f"{relation}-{entities}"), 'w')
    writer = csv.writer(out_file, delimiter='\t')
    file_paths.sort()
    positive_skipped = set()
    rows_used_per_pattern = {}
    for i, relation_path in enumerate(file_paths):
        rows_used = 0
        search_file = open(relation_path, "r", encoding="utf-8")
        print(f"Working on {relation_path}")
        reader = csv.reader(search_file, delimiter='\t')
        headers = next(reader)
        for d in tqdm(reader):
            d = map_array_given_header(d, headers)
            if d['sentence_id'] in positive_sent_ids_used:
                positive_skipped.add(d['sentence_id'])
                continue
            if not seperate_entities(d) or d['sentence_id'] == last_sent_id_used:
                continue
            # entities are not sorted in the same way all the time:
            if i==0: first_entity='e1'; second_entity='e2'
            elif i==1: first_entity='e2'; second_entity='e1'
            text = wrap_text(d['sentence_text'].split(),
                             d[f'{first_entity}_first_index'],
                             d[f'{first_entity}_last_index'] + 1,
                             d[f'{second_entity}_first_index'],
                             d[f'{second_entity}_last_index'] + 1)
            if dataset == 'docred':
                text = clean_special_tokens(text)

            writer.writerow([text, 'NOTA', NEGATIVE_PATTERNS[entities][i], d['sentence_id']])
            last_sent_id_used = d['sentence_id']
            rows_used += 1
        rows_used_per_pattern[i] = rows_used
        search_file.close()
    out_file.close()
    print(f"number of examples skipped because are positive: {len(positive_skipped)}")
    print(f"length positives: {len(set(positive_sent_ids_used))}")

    return rows_used_per_pattern

def map_array_given_header(arr, headers):
    def int_if_possible(value):
        try:
            int(value)
            return int(value)
        except ValueError:
            return value

    return {headers[i]: int_if_possible(arr[i]) for i in range(len(headers))}

def query_params(pattern, odinson):
    if odinson == False:
        return {
            "queries": {
                "syntactic": pattern
            },
            "data_set_name": "wikipedia",
            "include_annotations": False
        }
    else:
        pattern, expansion = pattern.split('#e ')
        return {
            "queries": {
                "odinson": pattern,
                "expansion": expansion
            },
            "data_set_name": "wikipedia",
            "include_annotations": False
            }

def download_from_spike_search(download_dir, patterns_dict, limit, use_odinson=False):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    outfiles = defaultdict(list)
    for relation, patterns in tqdm(patterns_dict.items()):
        for id, pattern in enumerate(patterns):
            search_query_api = '/api/3/search/query'
            search_query_params = query_params(pattern, use_odinson)
            download_tsv_params = f"?sentence_id=true&sentence_text=true&capture_indices=true"
            if limit > 0:
                download_tsv_params += f"&limit={limit}"

            print(f'Downloading query: {pattern} for relation: {relation}')
            request = requests.post(url=URL + search_query_api,
                                    headers={"Content-Type": "application/json"},
                                    data=json.dumps(search_query_params))

            tsv_location = request.headers['TSV-Location']
            tsv_url = URL + tsv_location + download_tsv_params

            outfile = f'{download_dir}/raw-{relation}-{id}'
            wget.download(tsv_url, outfile, bar=None)
            lines_downloaded = sum(1 for line in open(outfile, 'r'))
            print(f'Done downloading. lines downloaded: {lines_downloaded-1}')
            outfiles[relation] += [outfile]

    return outfiles

def clean_special_tokens(text_str):
    CLEANINGMAP = {'-RRB-': ')', '-LRB-': '(', '-LSB-': '[',
                   '-RSB-': ']', '-LCB-': '{', '-RCB-': '}',
                   '&nbsp;': ' ', '&quot;': "'", '--': '-', '---': '-'}
    tokens = text_str.split(' ')
    return ' '.join([CLEANINGMAP.get(t, t) for t in tokens])

def update_file_lengths(file_path, relations_num_rows):
    if not os.path.exists(file_path):
        lengths = relations_num_rows
    else:
        lengths = json.load(open(file_path, 'r'))
        for k, v in relations_num_rows.items():
            lengths[k] = v

    with open(file_path, 'w') as file:
        json.dump(lengths, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triggers", type=str, required=True, choices=['all', 'single'])
    parser.add_argument("--dataset", type=str, required=True, choices=['tacred', 'docred'])
    parser.add_argument("--download", action='store_true')
    parser.add_argument("--merge_patterns", action='store_true')
    args = parser.parse_args()
    main(args)