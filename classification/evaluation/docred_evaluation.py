#!/usr/bin/env python
import argparse
import os
import os.path
import json

from classification.docred_config import RELATION_MAPPING
from classification.docred import DocREDUtils

def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train

def correct_entity_types(relation_object, entities, relation_name):
    def get_entity_type(side: str):
        return entities[relation_object[side]][0]['type']

    return get_entity_type('h') in RELATION_MAPPING[relation_name]['e1_type'] and \
            get_entity_type('t') in RELATION_MAPPING[relation_name]['e2_type']

def main(args):
    relation_id = RELATION_MAPPING[args.relation_name]['id']

    truth_file = os.path.join(args.gold_dir, args.gold_file)
    truth = json.load(open(truth_file))

    std = {}
    std_in_single_sent = {}
    tot_evidences = 0
    titleset = set([])


    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']

        for label in x['labels']:
            r = label['r']

            h_idx = label['h']
            t_idx = label['t']
            if r != relation_id: continue
            if not correct_entity_types(label, vertexSet, args.relation_name): continue

            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])
            if len(label['evidence']) == 1 and len(DocREDUtils.evidences_with_entities(x, label)) > 0:
                std_in_single_sent[(title, r, h_idx, t_idx)] = set(label['evidence'])

    submission_answer_file = args.pred_file
    tmp = json.load(open(submission_answer_file))
    if len(tmp) == 0:
        if args.output_file:
            json.dump({
                "F1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "best_F1": 0.0,
                "best_precision": 0.0,
                "best_recall": 0.0,
                "best_confidence": (0.5 - 1e-10)},
                      open(args.output_file, 'w'))
        return

    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i-1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    submission_answer = sorted(submission_answer, key=lambda x: x['c'], reverse=True)

    if len(set([answer['r'] for answer in submission_answer])) != 1:
        raise ValueError('Mutliple relation predictions are passed')
        # This is a must as we are only adding a the "relation_name" to the std dict

    scores = eval(args, submission_answer, std_in_single_sent)
    # multi_sent_rel_scores = eval(args, submission_answer, std)

    # for k, v in multi_sent_rel_scores.items():
    #     scores[f"multi_sent_{k}"] = v

    if args.output_file:
        json.dump(scores, open(args.output_file, 'w'))

    return scores

def eval(args, submission_answer, std):
    correct_re = 0
    tot_relations = len(std)

    re_f1, re_p, re_r, best_f1, best_p, best_r, best_confidence = 0, 0, 0, 0, 0, 0, (0.5 - 1e-10)
    for i, x in enumerate(submission_answer):
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        confidence = x['c']
        if confidence < args.confidence_threshold:
            break

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1

        re_p = 1.0 * correct_re / (i+1)
        re_r = 1.0 * correct_re / tot_relations

        if re_p+re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        if best_f1 < re_f1:
            best_f1 = re_f1
            best_p = re_p
            best_r = re_r
            best_confidence = confidence

    scores = {
        "F1": re_f1,
        "precision": re_p,
        "recall": re_r,
        "best_F1": best_f1,
        "best_precision": best_p,
        "best_recall": best_r,
        "best_confidence": best_confidence
    }

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gold_dir', '--gold_dir',
        type=str,
        required=True)
    parser.add_argument('-gold_file', '--gold_file',
        type=str,
        required=True)
    parser.add_argument('-pred_file', '--pred_file',
        type=str,
        required=True)
    parser.add_argument('-output_file', '--output_file',
        default='evaluation',
        type=str)
    parser.add_argument('-relation_name', '--relation_name',
        type=str,
        required=True)
    parser.add_argument('-confidence_threshold', '--confidence_threshold',
        type=float,
        default=0)
    args = parser.parse_args()

    main(args)