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

TACRED_DOCRED_RELATIONS_MAPPING = {
    "per:children": "child",
    "per:date_of_birth": "date_of_birth",
    "org:dissolved": "dissolved,_abolished_or_demolished",
    "org:founded_by": "founded_by",
    "org:country_of_headquarters": "headquarters_location",
    "per:country_of_birth": "place_of_birth",
    "per:religion": "religion",
    "per:spouse": "spouse",
    "per:origin": "country_of_origin",
    "NOTA": "NOTA"
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
        "{e1:e=ORGANIZATION Microsoft} announced [t:w=extradition|bankruptcy|bankrupcy bankruptcy] in {e2:e=DATE 1997}.",
        ],
    "org:founded_by": [
        "{e2:e=PERSON Mary} [t:w=founder|co-founder|cofounder|creator founder] of {e1:e=ORGANIZATION Microsoft} likes running.",
        "{e2:e=PERSON Mary} , who [t:w=craft|crafted|crafts|crafting|create|creates|co-founded|co-found|created|creating|creation|debut|dominated|dominates|dominating|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formation|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|organize|organizes|organizing|organized|shapes|shaped|shaping|start|started|starting|starts founded] {e1:e=ORGANIZATION Microsoft} was thirsty.",
        "{e1:e=ORGANIZATION Microsoft} was [t:w=craft|crafted|crafts|crafting|create|creates|co-founded|co-found|created|creating|creation|debut|dominated|dominates|dominating|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formation|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|organize|organizes|organizing|organized|shapes|shaped|shaping|start|started|starting|starts founded] [$ by] {e2:e=PERSON Mary}.",
        ],
    "org:country_of_headquarters": [
        "{e1:e=ORGANIZATION Microsoft} is [t:w=based|headquarter|headquartered|headquarters|base based] in {e2:e=LOCATION England} .",
        "{e1:e=ORGANIZATION Microsoft} is [t:w=based|headquarter|headquartered|headquarters|base based] in {city:e=LOCATION London} , {e2:e=LOCATION England} .",
        "{e1:e=ORGANIZATION Microsoft} have [t:w=based|headquarter|headquartered|headquarters|base headquarters] in {e2:e=LOCATION England} .",
        ],
    "per:country_of_birth": [
        "{e1:e=PERSON John} was [t:w=born born] in {e2:e=LOCATION England} in 1997.",
        "{e1:e=PERSON John} was [t:w=born born] in {city:e=LOCATION London} , {e2:e=LOCATION England} in 1997.",
        "{e2:e=LOCATION England} [t:w=born born] {e1:e=PERSON John} is thirsty .",
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
        "{e1:e=PERSON Mary} is originally from {e2:e=LOCATIION Scotland}." #Check what's the best from these
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
    'PERSON:LOCATIION': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=LOCATIION]+) #e e1 e2", "(?<e1> [entity=LOCATIION]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
}


LIMIT = 5
OUTPUT_DIR = 'scripts/search_results'

def main():
    positive_outfiles = download_from_spike_search(PATTERNS, LIMIT)
    negative_outfiles = download_from_spike_search(NEGATIVE_PATTERNS, LIMIT*30, use_odinson=True)
    search_files = merge_files(positive_outfiles, negative_outfiles)
    prepare_examples(search_files)

def merge_files(positives, negatives):
    all = positives.copy()
    for k in all:
        all[k] += negatives[RELATIONS_TYPES[k]]
    return all

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

def prepare_examples(search_files): #TODO required better name
    max_positive_examples = 10000

    for relation, relation_search_files in search_files.items():
        all_data = []
        for i, search_file in enumerate(relation_search_files):
            data = read_tsv(search_file)
            for d in data:
                if not seperate_entities(d):
                    continue
                text = wrap_text(d['sentence_text'].split(),
                                d['e1_first_index'],
                                d['e1_last_index'] + 1,
                                d['e2_first_index'],
                                d['e2_last_index'] + 1)
                if relation in search_file:
                    label = relation
                    pattern = PATTERNS[relation][i]
                else:
                    label = 'NOTA'
                    pattern = 'Negative Example'
                all_data.append({'text': text, 'label': label, 'pattern': pattern, 'sentence_id': d['sentence_id']})

        all_data = remove_same_sent_id(all_data)
        all_data = sample(all_data, max_positive_examples, 10)

        write_to_datasets(all_data, relation)
        
def write_to_datasets(data, relation):
    for dataset in ['tacred', 'docred']:
        relation = relation if dataset == 'tacred' else TACRED_DOCRED_RELATIONS_MAPPING[relation]
        output_dataset_dir = os.path.join('data', dataset, 'search')
        if not os.path.exists(output_dataset_dir):
            os.makedirs(output_dataset_dir)
        with open(os.path.join(output_dataset_dir, relation), 'w') as outfile:
            tsv_writer = csv.writer(outfile, delimiter='\t')
            for i, d in enumerate(data):
                label = d['label'] if dataset == 'tacred' else TACRED_DOCRED_RELATIONS_MAPPING[d['label']]
                tsv_writer.writerow([i, d['text'], label, d['pattern'], d['sentence_id']])
            print(f"Wrote data to file {outfile.name}")

#Duplicated from re_processors
def sample(data,
           num_positive,
           negative_ratio):
    def get_first_num_examples(examples_in_label, max_num):
        if max_num is None:
            return examples_in_label
        return examples_in_label[:max_num]

    shuffle(data)
    positive_examples = get_first_num_examples([d for d in data if d['label'] != 'NOTA'], num_positive)
    negative_examples = get_first_num_examples([d for d in data if d['label'] == 'NOTA'], len(positive_examples) * negative_ratio)
    pos_and_neg_examples = positive_examples + negative_examples
    shuffle(pos_and_neg_examples)

    return pos_and_neg_examples

def read_tsv(input_file):
    def int_if_possible(value):
        try:
            int(value)
            return int(value)
        except ValueError:
            return value
    lines = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            lines.append(row)

    headers, data = lines[0], lines[1:]
    ret = []
    for d in data:
        ret.append(
            {headers[i]: int_if_possible(d[i]) for i in range(len(headers))}
        )

    return ret

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
            url = 'http://35.246.149.250:5000'
            search_query_api = '/api/3/search/query'
            search_query_params = query_params(pattern, use_odinson)
            download_tsv_params = f"?limit={limit}&sentence_id=true&sentence_text=true&capture_indices=true"

            request = requests.post(url=url + search_query_api,
                                    headers={"Content-Type": "application/json"},
                                    data=json.dumps(search_query_params))
            
            tsv_location = request.headers['TSV-Location']
            tsv_url = url + tsv_location + download_tsv_params
  
            print(f'Downloading query: {pattern} for relation: {relation}')
            outfile = f'{OUTPUT_DIR}/raw-{relation}-{id}'
            wget.download(tsv_url, outfile, bar=None)
            print('Done downloading ')
            outfiles[relation] += [outfile]

    return outfiles

#TODO, some of the results returns low numbers, redo them.

if __name__ == "__main__":
    main()
