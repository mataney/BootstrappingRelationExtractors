import argparse
from copy import deepcopy
import json

class ExploreDocRED:
    @staticmethod
    def print_false_negatives_adapter(preds, gold):
        print("\n########## FALSE NEGATIVES ##########\n")
        for pred in preds:
            doc = [g for g in gold if g['title'] == pred[0]][0]
            head = doc['vertexSet'][pred[1]]
            tail = doc['vertexSet'][pred[2]]
            head_appears_in = [o['sent_id'] for o in head]
            tail_appears_in = [o['sent_id'] for o in tail]

            evidence = list(set(head_appears_in) & set(tail_appears_in))
            if len(evidence) == 0:
                print("No overlapping sents between entities")
                continue
            if len(evidence) > 1:
                print("can't be sure of the correct sentence: ", end='')
            print(ExploreDocRED.colorify(deepcopy(doc['sents']), head, tail, evidence))

    @staticmethod
    def print_false_positive_adapter(preds, gold):
        print("\n########## FALSE POSITIVES ##########\n")
        for pred in preds:
            doc = [g for g in gold if g['title'] == pred['title']][0]
            head = doc['vertexSet'][pred['h_idx']]
            tail = doc['vertexSet'][pred['t_idx']]
            head_appears_in = [o['sent_id'] for o in head]
            tail_appears_in = [o['sent_id'] for o in tail]

            evidence = list(set(head_appears_in) & set(tail_appears_in))
            if len(evidence) == 0:
                print("No overlapping sents between entities")
                continue
            if len(evidence) > 1:
                print("can't be sure of the correct sentence: ", end='')
            print(ExploreDocRED.colorify(deepcopy(doc['sents']), head, tail, evidence))

    @staticmethod
    def print_true_pos_adapter(preds, gold):
        print("\n########## CORRECT ##########\n")
        for pred in preds:
            doc = [g for g in gold if g['title'] == pred['title']][0]
            head = doc['vertexSet'][pred['h_idx']]
            tail = doc['vertexSet'][pred['t_idx']]
            label = [label for label in doc['labels'] if label['h'] == pred['h_idx'] and label['t'] == pred['t_idx'] and label['r'] == pred['r']][0]
            evidence = label['evidence']
            if len(evidence) > 1:
                print("can't be sure of the correct sentence: ", end='')
            print(ExploreDocRED.colorify(deepcopy(doc['sents']), head, tail, evidence))

    @staticmethod
    def get_false_negatives(preds, relevant_gold, confidence_threshold):
        false_negatives = []
        preds_to_conf = {(p['title'], p['h_idx'], p['t_idx'], p['r']): p['c'] for p in preds}
        for doc in relevant_gold.values():
            doc_in_pred_style = (doc['title'], doc['label']['h'], doc['label']['t'], doc['label']['r'])
            if doc_in_pred_style not in preds_to_conf or preds_to_conf[doc_in_pred_style] < confidence_threshold:
                false_negatives.append(doc_in_pred_style)
        return false_negatives

    @staticmethod
    def get_true_pos_and_false_pos(preds, confidence_threshold, relevant_gold):
        correct, false_positives = [], []
        for pred in preds:
            if pred['c'] < confidence_threshold:
                continue
            title = pred['title']
            h_idx = pred['h_idx']
            t_idx = pred['t_idx']
            if (title, h_idx, t_idx) in relevant_gold:
                correct.append(pred)
            else:
                false_positives.append(pred)
        
        return correct, false_positives

    # Method taken from evlaute.py
    @staticmethod
    def remove_duplicates(preds):
        without_duplicates = [preds[0]]
        for i in range(1, len(preds)):
            x = preds[i]
            y = preds[i-1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                without_duplicates.append(preds[i])

        return without_duplicates

    @staticmethod
    def populate_relevant_gold(gold, relevant_relation):
        relevant_golds = {}
        for doc in gold:
            for label in doc['labels']:
                if label['r'] == relevant_relation:
                    title = doc['title']
                    key = (title, label['h'], label['t'])
                    relevant_golds[key] = {'vertexSet': doc['vertexSet'], 'title': doc['title'], 'sents': doc['sents'], 'label': label}
        return relevant_golds

    @staticmethod
    def colorify(text, head, tail, evidence):
        colors = dict(
            red='\x1b[31m',
            green='\x1b[32m',
            orange='\x1b[33m',
            blue='\x1b[34m',
            purple='\x1b[35m',
            back_to_white="\x1b[00m")

        entities = []
        for h in head:
            entities.append(h)
            entities[-1].update({'type': 'head'})

        for t in tail:
            entities.append(t)
            entities[-1].update({'type': 'tail'})

        entities = sorted(entities, key=lambda x: (x['sent_id'], x['pos'][0]), reverse=True)

        for ent in entities:
            color = 'red' if ent['type'] == 'head' else 'orange'
            text[ent['sent_id']].insert(ent['pos'][-1], colors['back_to_white'])
            text[ent['sent_id']].insert(ent['pos'][0], colors[color])

        text = f" {colors['green']}<ENDSENT>{colors['back_to_white']} ".join([' '.join(t) for i, t in enumerate(text) if i in evidence])
        for color in colors.values():
            text = text.replace(f" {color} ", f"{color} ")

        return text

class ExploreTACRED:
    @staticmethod
    def print_false_negatives_adapter(preds, gold):
        print("\n########## FALSE NEGATIVES ##########\n")
        for pred in preds:
            head_range = (pred['subj_start'], pred['subj_end']+1)
            tail_range = (pred['obj_start'], pred['obj_end']+1)

            text = pred['token']
            print(ExploreTACRED.colorify(deepcopy(text), head_range, tail_range))

    @staticmethod
    def print_positives_adapter(preds, gold):
        for pred in preds:
            corresponding_gold = gold[pred['title']]
            head_range = (corresponding_gold['subj_start'], corresponding_gold['subj_end']+1)
            tail_range = (corresponding_gold['obj_start'], corresponding_gold['obj_end']+1)

            text = corresponding_gold['token']
            print(ExploreTACRED.colorify(deepcopy(text), head_range, tail_range))

    @staticmethod
    def print_false_positive_adapter(preds, gold):
        print("\n########## FALSE POSITIVES ##########\n")
        ExploreTACRED.print_positives_adapter(preds, gold)

    @staticmethod
    def print_true_pos_adapter(preds, gold):
        print("\n########## CORRECT ##########\n")
        ExploreTACRED.print_positives_adapter(preds, gold)

    @staticmethod
    def get_false_negatives(preds, relevant_gold, confidence_threshold):
        false_negatives = []
        preds_to_conf = {p['title']: p['c'] for p in preds}
        for id, doc in relevant_gold.items():
            if id not in preds_to_conf or preds_to_conf[id] < confidence_threshold:
                false_negatives.append(doc)
        return false_negatives

    @staticmethod
    def get_true_pos_and_false_pos(preds, confidence_threshold, relevant_gold):
        correct, false_positives = [], []
        for pred in preds:
            if pred['c'] < confidence_threshold:
                continue
            title = pred['title']
            if title in relevant_gold:
                correct.append(pred)
            else:
                false_positives.append(pred)
        
        return correct, false_positives

    # Method taken from evlaute.py
    @staticmethod
    def remove_duplicates(preds):
        without_duplicates = [preds[0]]
        for i in range(1, len(preds)):
            x = preds[i]
            y = preds[i-1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                without_duplicates.append(preds[i])

        return without_duplicates

    @staticmethod
    def populate_relevant_gold(gold, relevant_relation):
        relevant_golds = {}
        for i, doc in enumerate(gold):
            if doc['relation'] == relevant_relation:
                relevant_golds[i] = doc
        return relevant_golds

    @staticmethod
    def colorify(text, head, tail):
        colors = dict(
            red='\x1b[31m',
            green='\x1b[32m',
            orange='\x1b[33m',
            blue='\x1b[34m',
            purple='\x1b[35m',
            back_to_white="\x1b[00m")

        entities = [{"type": "head", 'pos': head},
                    {"type": "tail", 'pos': tail},]
        entities = sorted(entities, key=lambda x: (x['pos'][0]), reverse=True)

        for ent in entities:
            color = 'red' if ent['type'] == 'head' else 'orange'
            text.insert(ent['pos'][-1], colors['back_to_white'])
            text.insert(ent['pos'][0], colors[color])

        text = " ".join(text)

        for color in colors.values():
            text = text.replace(f" {color} ", f"{color} ")

        return text


def main(preds, gold, args):

    explorer = ExploreTACRED if args.dataset == 'tacred' else ExploreDocRED

    relevant_relation = preds[0]['r']
    preds = explorer.remove_duplicates(preds)
    relevant_gold = explorer.populate_relevant_gold(gold, relevant_relation)

    true_pos, false_positives = explorer.get_true_pos_and_false_pos(preds, args.confidence_threshold, relevant_gold)
    false_negatives = explorer.get_false_negatives(preds, relevant_gold, args.confidence_threshold)
    
    assert len(true_pos) + len(false_negatives) == len(relevant_gold)
    
    if 'tp' in args.report:
        explorer.print_true_pos_adapter(true_pos, gold)
    if 'fp' in args.report:
        explorer.print_false_positive_adapter(false_positives, gold)
    if 'fn' in args.report:
        explorer.print_false_negatives_adapter(false_negatives, gold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', choices=['tacred', 'docred'], default='tacred')
    parser.add_argument('-raw', '--raw',
        type=str,
        required=True)
    parser.add_argument('-confidence_threshold', '--confidence_threshold', type=float, required=True)
    parser.add_argument('-report', '--report', nargs='+', choices=['tp', 'fp', 'fn'], default='tp fp fn')
    args = parser.parse_args()
    
    with open(args.raw, 'r') as f:
        raw = json.load(f)

    test_set = 'test' if args.dataset == 'tacred' else 'dev'
    with open(f'../data/{args.dataset}/{test_set}.json', 'r') as f:
        gold = json.load(f)

    main(raw['full_test_eval_results'][0], gold, args)