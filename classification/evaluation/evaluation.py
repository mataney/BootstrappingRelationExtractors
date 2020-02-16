#!/usr/bin/env python
import argparse
import os
import os.path
import json

from classification.docred_config import CLASS_MAPPING

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

def main(args):
    relation_name = CLASS_MAPPING[args.relation_name]['id']
    if args.ignore_train:
        fact_in_train_annotated = gen_train_facts("train_annotated.json", args.gold_dir)
        fact_in_train_distant = gen_train_facts("train_distant.json", args.gold_dir)

    truth_file = os.path.join(args.gold_dir, args.gold_file)
    truth = json.load(open(truth_file))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']

            h_idx = label['h']
            t_idx = label['t']
            if r == relation_name:
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

    tot_relations = len(std)

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
                "best_confidence": 0.0},
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
        raise ValueError('Mutliple relatoin predictions are passed')
        # This is a must as we are only adding a the "relation_name" to the std dict

    correct_re = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    re_f1, re_p, re_r, best_f1, best_p, best_r, best_confidence = 0, 0, 0, 0, 0, 0, 1.0
    for i, x in enumerate(submission_answer):
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        confidence = x['c']
        if confidence < args.confidence_threshold:
            break
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            in_train_annotated = in_train_distant = False
            if args.ignore_train:
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1

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

        if args.ignore_train:
            re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)
            re_p_ignore_train = 1.0 * (correct_re-correct_in_train_distant) / (len(submission_answer)-correct_in_train_distant)

            if re_p_ignore_train_annotated+re_r == 0:
                re_f1_ignore_train_annotated = 0
            else:
                re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

            if re_p_ignore_train+re_r == 0:
                re_f1_ignore_train = 0
            else:
                re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)


    scores = {
        "F1": re_f1,
        "precision": re_p,
        "recall": re_r,
        "best_F1": best_f1,
        "best_precision": best_p,
        "best_recall": best_r,
        "best_confidence": best_confidence
    }

    if args.output_file:
        json.dump(scores, open(args.output_file, 'w'))

    if args.ignore_train:
        print ('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)
        print ('RE_ignore_distant_F1:', re_f1_ignore_train)

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
    parser.add_argument('-ignore_train', '--ignore_train',
        action="store_true",
        help="in the original script they also report F1 if you exclude pairs that were in the training sets")
    args = parser.parse_args()

    main(args)