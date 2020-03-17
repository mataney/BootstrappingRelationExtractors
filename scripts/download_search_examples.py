from collections import defaultdict
import csv
import json
import os
import requests
from tqdm import tqdm
from random import shuffle
import wget

from classification.re_processors import wrap_text

RELATIONS_TYPES = {
    "per:children": "PERSON:PERSON",
    "per:date_of_birth": "PERSON:DATE",
    "org:dissolved": "ORGANIZATION:DATE",
    "org:founded_by": "ORGANIZATION:PERSON",
    "org:country_of_headquarters": "ORGANIZATION:LOCATION",
    "per:country_of_birth": "PERSON:LOCATION",
    "per:religion": "PERSON:MISC",
    "per:spouse": "PERSON:PERSON",
    "per:origin": "PERSON:MISC",
}

SINGLE_TRIGGER_PATTERNS = {
    "per:children": [
        "{e1:e=PERSON John} 's [t:w=son son] , {e2:e=PERSON Tim} .",
        "{e1:e=PERSON John} is survived by her [t:w=son son] , {e2:e=PERSON Tim} .",
        "{e1:e=PERSON Mary} gave [t:w=birth birth] [$ to] {e2:e=PERSON John}",
        ],
    "per:date_of_birth": [
        "{e1:e=PERSON John} was [t:w=born born] in {e2:e=DATE 1997} .",
        "{e1:e=PERSON John} was [t:w=born born] in San Francisco in {e2:e=DATE 1997}",
        "{e1:e=PERSON John} [$ -LRB-] {e2:e=DATE 1997} [$ -] [$:e=DATE date] [$ -RRB-] .",
        ],
    "org:dissolved": [
        "{e1:e=ORGANIZATION Microsoft} was [t:w=closed closed] in {e2:e=DATE 1997} .",
        "{e1:e=ORGANIZATION Microsoft} announced [t:w=bankruptcy bankruptcy] in {e2:e=DATE 1997}.",
        "{e1:e=ORGANIZATION Microsoft} filed for [t:w=bankruptcy bankruptcy] in {e2:e=DATE 1997}. ",
        ],
    "org:founded_by": [
        "{e1:e=ORGANIZATION Microsoft} [t:w=founder founder] {e2:e=PERSON Mary} likes running.",
        "{e2:e=PERSON Mary} , who [t:w=founded founded] {e1:e=ORGANIZATION Microsoft} was thirsty.",
        "{e1:e=ORGANIZATION Microsoft} was [t:w=founded founded] [$ by] {e2:e=PERSON Mary}.",
        ],
    "org:country_of_headquarters": [
        "{e1:e=ORGANIZATION Microsoft} is [t:w=based based] in {e2:e=LOCATION England} .",
        "{e1:e=ORGANIZATION Microsoft} is [t:w=based based] in {city:e=LOCATION London} , {e2:e=LOCATION England} .",
        "{e1:e=ORGANIZATION Microsoft}, [t:w=based based] in {city:e=LOCATION London} , {e2:e=LOCATION England} .",
        ],
    "per:country_of_birth": [
        "{e1:e=PERSON John} was [t:w=born born] in {e2:e=LOCATION England} in 1997.",
        "{e1:e=PERSON John} was [t:w=born born] in {city:e=LOCATION London} , {e2:e=LOCATION England} in 1997.",
        "{e1:e=PERSON John} [$ -LRB-] [t:w=born born] in Bremen, {e2:e=LOCATION Germany} [$ -RRB-] .",
        ],
    # "per:religion": [
    #     "{e1:e=PERSON John} is a [e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Conservative|Islamist|Islam|Muslim|Shia Jewish]",
    #     "[e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Conservative|Islamist|Islam|Muslim|Shia Jewish] {e1:e=PERSON John} is walking down the street.",
    #     ],
    "per:spouse": [
        "{e1:e=PERSON John} 's [t:w=wife wife], {e2:e=PERSON Mary} , died in 1991 .",
        "{e1:e=PERSON John} [t:w=married married] {e2:e=PERSON Mary}",
        "{e1:e=PERSON John} [t:w=married married] to {e2:e=PERSON Mary}",
        ],
    "per:origin": [
        "{e2:e=MISC Scottish} {e1:e=PERSON Mary} is high.",
        "{e1:e=PERSON Mary} is of {e2:e=MISC Scottish} descent.",
        "{e1:e=PERSON Mary} is of {e2:e=MISC Scottish} [t:w=descent|nationality|ancestry|heritage|roots|blood|maternal|birth|descends|paternal|descended|raised|born|background|descend|origins|lineage|origin|ancestors|descendant|ancestral|country descent].",
        "{e1:e=PERSON Mary} is originally from {e2:e=LOCATIION Scotland}." #Check what's the best from these
        ]
    }

PATTERNS = {
    "per:children": [
        "{e1:e=PERSON John} 's [t:w=baby|child|children|daughter|daughters|son|sons|step-daughter|step-son|step-child|step-children|stepchildren|stepdaughter|stepson child] , {e2:e=PERSON Mary} .",
        "{e1:e=PERSON John} is survived by her [t:w=baby|child|children|daughter|daughters|son|sons|step-daughter|step-son|step-child|step-children|stepchildren|stepdaughter|stepson child] , {e2:e=PERSON Mary} .",
        "{e1:e=PERSON Mary} gave [t:w=birth birth] [$ to] {e2:e=PERSON John}",
        ],
    "per:date_of_birth": [
        "{e1:e=PERSON John} was [t:w=born born] in {e2:e=DATE 1997} .",
        "{e1:e=PERSON John} was [t:w=born born] in San Francisco in {e2:e=DATE 1997}",
        "{e1:e=PERSON John} [$ -LRB-] {e2:e=DATE 1997} [$ -] [$:e=DATE date] [$ -RRB-] .",
        ],
    "org:dissolved": [
        "{e1:e=ORGANIZATION Microsoft} was [t:w=bust|closed|expired|dissolved|disbanded|bankrupted|dismantled|crumbled|ceased|collapsed closed] in {e2:e=DATE 1997} .",
        "{e1:e=ORGANIZATION Microsoft} announced [t:w=extradition|bankruptcy|bankrupcy|liquidation bankruptcy] in {e2:e=DATE 1997}.",
        "{e1:e=ORGANIZATION Microsoft} filed for [t:w=extradition|bankruptcy|bankrupcy|liquidation bankruptcy] in {e2:e=DATE 1997}. ",
        ],
    "org:founded_by": [
        "{e1:e=ORGANIZATION Microsoft} [t:w=founder|co-founder|cofounder|creator founder] {e2:e=PERSON Mary} likes running.",
        "{e2:e=PERSON Mary} , who [t:w=craft|crafted|crafts|crafting|create|creates|co-founded|co-found|created|creating|creation|debut|dominated|dominates|dominating|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formation|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|organize|organizes|organizing|organized|shapes|shaped|shaping|start|started|starting|starts founded] {e1:e=ORGANIZATION Microsoft} was thirsty.",
        "{e1:e=ORGANIZATION Microsoft} was [t:w=craft|crafted|crafts|crafting|create|creates|co-founded|co-found|created|creating|creation|debut|dominated|dominates|dominating|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formation|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|organize|organizes|organizing|organized|shapes|shaped|shaping|start|started|starting|starts founded] [$ by] {e2:e=PERSON Mary}.",
        ],
    "org:country_of_headquarters": [
        "{e1:e=ORGANIZATION Microsoft} is [t:w=based|headquarter|headquartered|headquarters|base based] in {e2:e=LOCATION England} .",
        "{e1:e=ORGANIZATION Microsoft} is [t:w=based|headquarter|headquartered|headquarters|base based] in {city:e=LOCATION London} , {e2:e=LOCATION England} .",
        "{e1:e=ORGANIZATION Microsoft}, [t:w=based|headquarter|headquartered|headquarters|base based] in {city:e=LOCATION London} , {e2:e=LOCATION England} .",
        ],
    "per:country_of_birth": [
        "{e1:e=PERSON John} was [t:w=born born] in {e2:e=LOCATION England} in 1997.",
        "{e1:e=PERSON John} was [t:w=born born] in {city:e=LOCATION London} , {e2:e=LOCATION England} in 1997.",
        "{e1:e=PERSON John} [$ -LRB-] [t:w=born born] in Bremen, {e2:e=LOCATION Germany} [$ -RRB-] .",
        ],
    "per:religion": [
        "{e1:e=PERSON John} is a [e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Conservative|Islamist|Islam|Muslim|Shia Jewish]",
        "[e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Conservative|Islamist|Islam|Muslim|Shia Jewish] {e1:e=PERSON John} is walking down the street.",
        ],
    "per:spouse": [
        "{e1:e=PERSON John} 's [t:w=ex-husband|ex-wife|husband|widow|widower|wife|sweetheart|bride wife], {e2:e=PERSON Mary} , died in 1991 .",
        "{e1:e=PERSON John} [t:w=divorce|divorced|married|marry|wed|divorcing married] {e2:e=PERSON Mary}",
        "{e1:e=PERSON John} [t:w=married|marry|wed married] to {e2:e=PERSON Mary}",
        ],
    "per:origin": [
        "{e2:e=MISC Scottish} {e1:e=PERSON Mary} is high.",
        "{e1:e=PERSON Mary} is of {e2:e=MISC Scottish} descent.",
        "{e1:e=PERSON Mary} is of {e2:e=MISC Scottish} [t:w=descent|nationality|ancestry|heritage|roots|blood|maternal|birth|descends|paternal|descended|raised|born|background|descend|origins|lineage|origin|ancestors|descendant|ancestral|country descent].",
        ]
    }

NEGATIVE_PATTERNS = {
    'PERSON:PERSON': ["(?<e1> [entity=PERSON]+) [entity!=PERSON]+ (?<e2> [entity=PERSON]+) #e e1 e2"],
    'PERSON:DATE': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=DATE]+) #e e1 e2", "(?<e1> [entity=DATE]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
    'ORGANIZATION:DATE': ["(?<e1> [entity=ORGANIZATION]+) []+ (?<e2> [entity=DATE]+) #e e1 e2", "(?<e1> [entity=DATE]+) []+ (?<e2> [entity=ORGANIZATION]+) #e e1 e2"],
    'ORGANIZATION:PERSON': ["(?<e1> [entity=ORGANIZATION]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2", "(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=ORGANIZATION]+) #e e1 e2"],
    'ORGANIZATION:LOCATION': ["(?<e1> [entity=ORGANIZATION]+) []+ (?<e2> [entity=LOCATION]+) #e e1 e2", "(?<e1> [entity=LOCATION]+) []+ (?<e2> [entity=ORGANIZATION]+) #e e1 e2"],
    'PERSON:LOCATION': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=LOCATION]+) #e e1 e2", "(?<e1> [entity=LOCATION]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
    'PERSON:MISC': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=MISC]+) #e e1 e2", "(?<e1> [entity=MISC]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
}


LIMIT = -1
OUTPUT_DIR = 'scripts/search_results'
URL = 'http://35.246.164.171:5000'

def main():
    # positive_outfiles = download_from_spike_search(PATTERNS, LIMIT)
    # negative_outfiles = download_from_spike_search(NEGATIVE_PATTERNS, LIMIT, use_odinson=True)
    positive_outfiles = {"org:country_of_headquarters": ["raw-org:country_of_headquarters-0", "raw-org:country_of_headquarters-1", "raw-org:country_of_headquarters-2"],
                         "org:dissolved": ["raw-org:dissolved-0", "raw-org:dissolved-1", "raw-org:dissolved-2"],
                         "org:founded_by": ["raw-org:founded_by-0", "raw-org:founded_by-1", "raw-org:founded_by-2"],
                         "per:children": ["raw-per:children-0", "raw-per:children-1", "raw-per:children-2"],
                         "per:country_of_birth": ["raw-per:country_of_birth-0", "raw-per:country_of_birth-1", "raw-per:country_of_birth-2"],
                         "per:date_of_birth": ["raw-per:date_of_birth-0", "raw-per:date_of_birth-1", "raw-per:date_of_birth-2"],
                         "per:origin": ["raw-per:origin-0", "raw-per:origin-1", "raw-per:origin-2"],
                         "per:religion": ["raw-per:religion-0", "raw-per:religion-1"],
                         "per:spouse": ["raw-per:spouse-0", "raw-per:spouse-1", "raw-per:spouse-2"],}
    negative_outfiles = {"ORGANIZATION:DATE": ["raw-ORGANIZATION:DATE-0", "raw-ORGANIZATION:DATE-1"],
                         "ORGANIZATION:LOCATION": ["raw-ORGANIZATION:LOCATION-0", "raw-ORGANIZATION:LOCATION-1"],
                         "ORGANIZATION:PERSON": ["raw-ORGANIZATION:PERSON-0", "raw-ORGANIZATION:PERSON-1"],
                         "PERSON:DATE": ["raw-PERSON:DATE-0", "raw-PERSON:DATE-1"],
                         "PERSON:LOCATION": ["raw-PERSON:LOCATION-0", "raw-PERSON:LOCATION-1"],
                         "PERSON:MISC": ["raw-PERSON:MISC-0", "raw-PERSON:MISC-1"],
                         "PERSON:PERSON": ["raw-PERSON:PERSON-0"],}
    output_dir = os.path.join('data', 'search')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    positive_length_counter = prepare_positive_examples(output_dir, positive_outfiles)
    negative_length_counter = prepare_negative_examples(output_dir, negative_outfiles)

    with open(os.path.join(output_dir, 'file_lengths.json'), 'w') as files_lengths:
        json.dump(dict(positive_length_counter, **negative_length_counter), files_lengths)


def remove_same_sent_id(data):
    grouped = defaultdict(list)
    for d in data:
        grouped[d['sentence_id']].append(d)

    ret = []
    for k, v in grouped.items():
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

def prepare_positive_examples(output_dir, search_file_paths):
    row_ids = {}
    for relation, relation_paths in tqdm(search_file_paths.items()):
        last_sent_id_used = -1
        out_file = open(os.path.join(output_dir, relation), 'w')
        writer = csv.writer(out_file, delimiter='\t')
        row_id = 0
        for i, relation_path in enumerate(relation_paths):
            relation_path = os.path.join('scripts', 'search_results', relation_path)
            search_file = open(relation_path, "r", encoding="utf-8")
            reader = csv.reader(search_file, delimiter='\t')
            headers = next(reader)
            for d in reader:
                d = map_array_given_header(d, headers)
                if not seperate_entities(d) or d['sentence_id'] == last_sent_id_used:
                    continue
                text = wrap_text(d['sentence_text'].split(),
                                 d['e1_first_index'],
                                 d['e1_last_index'] + 1,
                                 d['e2_first_index'],
                                 d['e2_last_index'] + 1)
                writer.writerow([text, relation, PATTERNS[relation][i], d['sentence_id']])
                last_sent_id_used = d['sentence_id']
                row_id += 1
            search_file.close()
        row_ids[relation] = row_id
        out_file.close()

    return row_ids

def prepare_negative_examples(output_dir, search_file_paths):
    row_ids = {}
    for entities, file_paths in tqdm(search_file_paths.items()):
        last_sent_id_used = -1
        out_file = open(os.path.join(output_dir, entities), 'w')
        writer = csv.writer(out_file, delimiter='\t')
        row_id = 0
        for i, relation_path in enumerate(file_paths):
            relation_path = os.path.join('scripts', 'search_results', relation_path)
            search_file = open(relation_path, "r", encoding="utf-8")
            reader = csv.reader(search_file, delimiter='\t')
            headers = next(reader)
            for d in reader:
                d = map_array_given_header(d, headers)
                if not seperate_entities(d) or d['sentence_id'] == last_sent_id_used:
                    continue
                text = wrap_text(d['sentence_text'].split(),
                                 d['e1_first_index'],
                                 d['e1_last_index'] + 1,
                                 d['e2_first_index'],
                                 d['e2_last_index'] + 1)
                writer.writerow([text, 'NOTA', NEGATIVE_PATTERNS[entities][i], d['sentence_id']])
                last_sent_id_used = d['sentence_id']
                row_id += 1
            search_file.close()
        row_ids[entities] =row_id
        out_file.close()

    return row_ids

# def write_to_datasets(data, relation):
#     for dataset in ['tacred', 'docred']:
#         relation = relation if dataset == 'tacred' else TACRED_DOCRED_RELATIONS_MAPPING[relation]
#         output_dataset_dir = os.path.join('data', dataset, 'search')
#         if not os.path.exists(output_dataset_dir):
#             os.makedirs(output_dataset_dir)
#         with open(os.path.join(output_dataset_dir, relation), 'w') as outfile:
#             writer = csv.writer(outfile, delimiter='\t')
#             for i, d in enumerate(data):
#                 label = d['label'] if dataset == 'tacred' else TACRED_DOCRED_RELATIONS_MAPPING[d['label']]
#                 writer.writerow([i, d['text'], label, d['pattern'], d['sentence_id']])
#             print(f"Wrote data to file {outfile.name}")

# #Duplicated from re_processors
# def sample(data,
#            num_positive,
#            negative_ratio):
#     def get_first_num_examples(examples_in_label, max_num):
#         if max_num is None:
#             return examples_in_label
#         return examples_in_label[:max_num]

#     shuffle(data)
#     positive_examples = get_first_num_examples([d for d in data if d['label'] != 'NOTA'], num_positive)
#     negative_examples = get_first_num_examples([d for d in data if d['label'] == 'NOTA'], len(positive_examples) * negative_ratio)
#     pos_and_neg_examples = positive_examples + negative_examples
#     shuffle(pos_and_neg_examples)

#     return pos_and_neg_examples

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

def download_from_spike_search(patterns_dict, limit, use_odinson=False):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    outfiles = defaultdict(list)
    for relation, patterns in tqdm(patterns_dict.items()):
        for id, pattern in enumerate(patterns):
            search_query_api = '/api/3/search/query'
            search_query_params = query_params(pattern, use_odinson)
            download_tsv_params = f"?sentence_id=true&sentence_text=true&capture_indices=true"
            if limit > 0:
                download_tsv_params += f"&limit={limit}"

            request = requests.post(url=URL + search_query_api,
                                    headers={"Content-Type": "application/json"},
                                    data=json.dumps(search_query_params))
            
            tsv_location = request.headers['TSV-Location']
            tsv_url = URL + tsv_location + download_tsv_params
  
            print(f'Downloading query: {pattern} for relation: {relation}')
            outfile = f'{OUTPUT_DIR}/raw-{relation}-{id}'
            wget.download(tsv_url, outfile, bar=None)
            print('Done downloading ')
            outfiles[relation] += [outfile]

    return outfiles

if __name__ == "__main__":
    main()