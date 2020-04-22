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
        text, subjects, objects, e3, e4 = find_subject_and_objects(line)
        e3 = [['x', o] for o in e3]
        e4 = [['y', o] for o in e4]

        subj_id, obj_id = sample(subjects, 1)[0], sample(objects, 1)[0]
        ents = []
        for ent in subjects:
            if ent == subj_id:
                ents.append(['s', ent])
            else:
                ents.append(['x', ent])
        for ent in objects:
            if ent == obj_id:
                ents.append(['o', ent])
            else:
                ents.append(['x', ent])
        new_annotation_lines.append(wrap_text(text, ents + e3 + e4))

    with open(args.in_file_path.split('.txt')[0]+'_new_wraps.txt', 'w') as outfile:
        for line in new_annotation_lines:
            outfile.write(line)

def find_subject_and_objects(line):
    last_found = None
    i = 0
    subjects, objects, e3, e4 = [], [], [], []
    while i < len(line):
        if line[i] == '[':
            if line[i+1] in ['s', 'o', 'x', 'y']:
                last_found = line[i+1] # 's' or 'o'
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