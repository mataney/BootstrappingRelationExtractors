#!/usr/bin/env python
"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import json
import os
import sys
from collections import Counter

NO_RELATION = "no_relation"
PRONOUNS = ["he", "she", "it", "me", "us", "you", "her", "him", "it", "them", "my", "our", "your", "her", "his", "their"]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('-gold_dir', '--gold_dir',
                        help='The gold relation dir; one relation per line',
                        required=True)
    parser.add_argument('-gold_file', '--gold_file',
                        help='The gold relation file; one relation per line',
                        required=True)
    parser.add_argument('-pred_file', '--pred_file',
                        help='A prediction file; one relation per line, in the same order as the gold file.',
                        required=True)
    parser.add_argument('-output_file', '--output_file',
                        required=True)
    parser.add_argument('-relation_name', '--relation_name',
                        help='The relation we are checking',
                        required=True)
    parser.add_argument('-confidence_threshold', '--confidence_threshold',
                        default=0.5 - 1e-10,
                        type=float,
                        required=False)
    parser.add_argument('-remove_pronouns', '--remove_pronouns',
                        action='store_true',
                        help="Not using this")
    args = parser.parse_args()
    return args

def has_pronouns(gold_dict):
    subj = gold_dict['token'][gold_dict['subj_start']:gold_dict['subj_end']+1]
    obj = gold_dict['token'][gold_dict['obj_start']:gold_dict['obj_end']+1]
    return (len(subj) == 1 and subj[0].lower() in PRONOUNS) or (len(obj) == 1 and obj[0].lower() in PRONOUNS)

def score(key, prediction, args):
    best_f1, best_confidence = 0, (0.5 - 1e-10)
    prediction = sorted(prediction, key=lambda x: x['c'], reverse=True)
    if args.remove_pronouns:
        prediction = [p for p in prediction if not has_pronouns(key[p['title']])]
        gold_in_label = sum([1 for k in key if k['relation'] == args.relation_name and not has_pronouns(k)])
    else:
        gold_in_label = sum([1 for k in key if k['relation'] == args.relation_name])
    pred_in_label = len(prediction)

    correct_by_relation = 0
    prec = 1.0
    recall = 0.0
    f1 = 0.0
    # Loop over the data to compute a score
    for i, pred in enumerate(prediction):
        id = pred['title']
        gold_dict = key[id]
        gold = gold_dict['relation']

        if pred['c'] < args.confidence_threshold:
            break
         
        if gold == args.relation_name:
            correct_by_relation += 1
        
        if pred_in_label > 0:
            prec = float(correct_by_relation) / (i+1)
        if gold_in_label > 0:
            recall = float(correct_by_relation) / float(gold_in_label)
        if prec + recall > 0.0:
            f1 = 2.0 * prec * recall / (prec + recall)

        if f1 >= best_f1:
            best_f1 = f1
            best_confidence = pred['c']
    
    scores = {
        "F1": f1,
        "precision": prec,
        "recall": recall,
        "best_confidence": best_confidence,
        "best_f1": best_f1,
    }
    json.dump(scores, open(args.output_file, 'w'))
    return prec, recall, f1

def read_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return list(json.load(f))

if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = read_json(os.path.join(args.gold_dir, args.gold_file))
    prediction = read_json(args.pred_file)

    # Score the predictions
    score(key, prediction, args)
