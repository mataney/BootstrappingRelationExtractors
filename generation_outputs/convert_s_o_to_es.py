import argparse
from random import shuffle
import re

START_E1 = '[E1]'
END_E1 = '[/E1]'
START_E2 = '[E2]'
END_E2 = '[/E2]'

def main(args):
    with open(args.in_file_path, 'r') as infile:
        lines = infile.readlines()

    new_annotation_lines = []
    for line in lines:
        last_found = None
        i = 0
        while i < len(line):
            if line[i] == '[':
                if line[i+1] == 's':
                    line = line[:i] + START_E1 + line[i+2:]
                    last_found = 's'
                    i += 5
                elif line[i+1] == 'o':
                    line = line[:i] + START_E2 + line[i+2:]
                    last_found = 'o'
                    i += 5
                continue
            
            if line[i] == ']':
                if last_found == 's':
                    line = line[:i] + f" {END_E1}" + line[i+1:]
                    last_found = None
                    i += 6
                elif last_found == 'o':
                    line = line[:i] + f" {END_E2}" + line[i+1:]
                    last_found = None
                    i += 6
                else:
                    print("that's a problem")
                continue
            i += 1
        new_annotation_lines.append(line)

    shuffle(new_annotation_lines)

    with open(args.out_file_path, 'w') as outfile:
        for line in new_annotation_lines:
            outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file_path", type=str, required=True)
    parser.add_argument("--out_file_path", type=str, required=True)
    args = parser.parse_args()
    main(args)