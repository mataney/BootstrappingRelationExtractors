import argparse
from copy import deepcopy
import json

def main(preds, gold, args):

    relevant_relation = preds[0]['r']
    preds = remove_duplicates(preds)
    preds = remove_non_confident(preds, args.confidence_threshold)
    relevant_gold = populate_relevant_gold(gold, relevant_relation)

    true_pos, false_positives = get_true_pos_and_false_pos(preds, relevant_gold)
    false_negatives = get_false_negatives(preds, relevant_gold)
    
    assert len(true_pos) + len(false_negatives) == len(relevant_gold)
    
    if 'tp' in args.report:
        print_true_pos_adapter(true_pos, gold)
    if 'fp' in args.report:
        print_false_positive_adapter(false_positives, gold)
    if 'fn' in args.report:
        print_false_negatives_adapter(false_negatives, gold)

def print_false_negatives_adapter(preds, gold):
    print("\n########## FALSE NEGATIVES ##########\n")
    for pred in preds:
        doc = [g for g in gold if g['title'] == pred[0]][0]
        head = doc['vertexSet'][pred[1]]
        tail = doc['vertexSet'][pred[2]]
        head_appears_in = [o['sent_id'] for o in head]
        tail_appears_in = [o['sent_id'] for o in tail]

        evidence = intersection(head_appears_in, tail_appears_in)
        if len(evidence) == 0:
            print("No overlapping sents between entities")
            continue
        print(colorify(deepcopy(doc['sents']), head, tail, evidence))

def print_false_positive_adapter(preds, gold):
    print("\n########## FALSE POSITIVES ##########\n")
    for pred in preds:
        doc = [g for g in gold if g['title'] == pred['title']][0]
        head = doc['vertexSet'][pred['h_idx']]
        tail = doc['vertexSet'][pred['t_idx']]
        head_appears_in = [o['sent_id'] for o in head]
        tail_appears_in = [o['sent_id'] for o in tail]

        evidence = intersection(head_appears_in, tail_appears_in)
        if len(evidence) == 0:
            print("No overlapping sents between entities")
            continue
        print(colorify(deepcopy(doc['sents']), head, tail, evidence))

def print_true_pos_adapter(preds, gold):
    print("\n########## CORRECT ##########\n")
    for pred in preds:
        doc = [g for g in gold if g['title'] == pred['title']][0]
        head = doc['vertexSet'][pred['h_idx']]
        tail = doc['vertexSet'][pred['t_idx']]
        head_appears_in = [o['sent_id'] for o in head]
        tail_appears_in = [o['sent_id'] for o in tail]
        evidence_from_examples = intersection(head_appears_in, tail_appears_in)

        label = [label for label in doc['labels'] if label['h'] == pred['h_idx'] and label['t'] == pred['t_idx'] and label['r'] == pred['r']][0]
        evidence_from_dataset = label['evidence']

        evidence = intersection(evidence_from_examples, evidence_from_dataset)
        print(colorify(deepcopy(doc['sents']), head, tail, evidence))

def get_false_negatives(preds, relevant_gold):
    false_negatives = []
    preds_to_conf = [(p['title'], p['h_idx'], p['t_idx'], p['r']) for p in preds]
    for doc in relevant_gold.values():
        doc_in_pred_style = (doc['title'], doc['label']['h'], doc['label']['t'], doc['label']['r'])
        if doc_in_pred_style not in preds_to_conf:
            false_negatives.append(doc_in_pred_style)
    return false_negatives

def get_true_pos_and_false_pos(preds, relevant_gold):
    true_positives, false_positives = [], []
    for pred in preds:
        title = pred['title']
        h_idx = pred['h_idx']
        t_idx = pred['t_idx']
        if (title, h_idx, t_idx) in relevant_gold:
            true_positives.append(pred)
        else:
            false_positives.append(pred)
    
    return true_positives, false_positives

# Method taken from evlaute.py
def remove_duplicates(preds):
    preds.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    without_duplicates = [preds[0]]
    for i in range(1, len(preds)):
        x = preds[i]
        y = preds[i-1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            without_duplicates.append(preds[i])

    return without_duplicates

def remove_non_confident(preds, confidence_threshold):
    ret = []
    for pred in preds:
        if pred['c'] >= confidence_threshold:
            ret.append(pred)

    return ret

def populate_relevant_gold(gold, relevant_relation):
    relevant_golds = {}
    for doc in gold:
        for label in doc['labels']:
            if label['r'] == relevant_relation:
                title = doc['title']
                key = (title, label['h'], label['t'])
                relevant_golds[key] = {'vertexSet': doc['vertexSet'], 'title': doc['title'], 'sents': doc['sents'], 'label': label}
    return relevant_golds

def colorify(text, head, tail, evidence):
    colors = dict(
        red='\x1b[31m',
        green='\x1b[32m',
        orange='\x1b[33m',
        blue='\x1b[34m',
        purple='\x1b[35m',
        back_to_white="\x1b[00m")

    ret = ""
    if len(evidence) > 1:
        ret += f"{colors['red']}can't be sure of the correct sentence: {colors['back_to_white']}"

    entities = []
    for h in head:
        entities.append(h)
        entities[-1].update({'type': 'head'})

    for t in tail:
        entities.append(t)
        entities[-1].update({'type': 'tail'})

    entities = sorted(entities, key=lambda x: (x['sent_id'], x['pos'][0]), reverse=True)

    for ent in entities:
        color = 'green' if ent['type'] == 'head' else 'orange'
        text[ent['sent_id']].insert(ent['pos'][-1], colors['back_to_white'])
        text[ent['sent_id']].insert(ent['pos'][0], colors[color])

    ret += f" {colors['red']}<ENDSENT>{colors['back_to_white']} ".join([' '.join(t) for i, t in enumerate(text) if i in evidence])
    for color in colors.values():
        ret = ret.replace(f" {color} ", f"{color} ")
        if ret.startswith(f"{color} "):
            ret = ret[:5] + ret[6:]

    return ret

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw', '--raw',
        type=str,
        required=True)
    parser.add_argument('-confidence_threshold', '--confidence_threshold', type=float, required=True)
    parser.add_argument('-report', '--report', nargs='+', choices=['tp', 'fp', 'fn'])
    args = parser.parse_args()
    
    with open(args.raw, 'r') as f:
        raw = json.load(f)

    with open('data/DocRED/dev.json', 'r') as f:
        gold = json.load(f)

    main(raw['full_dev_eval_results'][0], gold, args)