import argparse
from itertools import product
from random import sample
from tqdm import tqdm

START_E1 = '[E1]'
END_E1 = '[/E1]'
START_E2 = '[E2]'
END_E2 = '[/E2]'
START_E3 = '[E3]'
END_E3 = '[/E3]'
START_E4 = '[E4]'
END_E4 = '[/E4]'

def main(args):
    with open(args.in_file_path, 'r') as infile:
        lines = infile.readlines()

    new_annotation_lines = []
    for i, line in tqdm(enumerate(lines)):
        assert line.count('[s') > 0 and line.count('[o') > 0, f"problem in line {i+1}"

        text, subjects, objects, e3, e4 = find_subject_and_objects(line)

        ents = mark_just_one_entity(subjects, 's', 'x')
        ents += mark_just_one_entity(objects, 'o', 'y')

        e3 = [['x', o] for o in e3]
        e4 = [['y', o] for o in e4]

        new_annotation_lines.append(wrap_text(text, ents + e3 + e4))

    with open(args.in_file_path.split('.txt')[0]+'_new_wraps.txt', 'w') as outfile:
        for line in new_annotation_lines:
            outfile.write(line)

def mark_just_one_entity(entities, pos_mark, neg_mark):
    entities = [[pos_mark, ent] for ent in entities]
    if len(entities) > 1:
        id_of_real_subj = sample(range(len(entities)), 1)[0]
        entities = [[pos_mark, ent[1]] if i == id_of_real_subj else [neg_mark, ent[1]] for i, ent in enumerate(entities)]
    return entities

def find_subject_and_objects(line):
    last_found = None
    i = 0
    subjects, objects, e3, e4 = [], [], [], []
    while i < len(line):
        if line[i] == '[':
            if line[i+1] in ['s', 'o', 'x', 'y']:
                last_found = line[i+1]
                last_found_index = i
                line = line[:i] + line[i+3:]
            continue

        if line[i] == ']':
            if last_found == 's':
                subjects.append((last_found_index, i))
                line = line[:i] + line[i+1:]
                last_found = None
            elif last_found == 'o':
                objects.append((last_found_index, i))
                line = line[:i] + line[i+1:]
                last_found = None
            elif last_found == 'x':
                e3.append((last_found_index, i))
                line = line[:i] + line[i+1:]
                last_found = None
            elif last_found == 'y':
                e4.append((last_found_index, i))
                line = line[:i] + line[i+1:]
                last_found = None
        i += 1
    return line, subjects, objects, e3, e4

def wrap_text(text, entities):
    entities = sorted(entities, key = lambda x: x[1][1], reverse=True)
    for ent in entities:
        if ent[0] == 's':
            start_symbol, end_symbol = START_E1, END_E1
        elif ent[0] == 'o':
            start_symbol, end_symbol = START_E2, END_E2
        if ent[0] == 'x':
            start_symbol, end_symbol = START_E3, END_E3
        if ent[0] == 'y':
            start_symbol, end_symbol = START_E4, END_E4
        text = text[:ent[1][0]] + f"{start_symbol} " + text[ent[1][0]: ent[1][1]] + f" {end_symbol}" + text[ent[1][1]:]
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file_path", type=str, required=True)
    args = parser.parse_args()
    main(args)