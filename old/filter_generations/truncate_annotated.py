import argparse
import os

from utils import read_file, write_to_file

def truncate(sentences):
    truncated = []
    for sent in sentences:
        

    return truncated

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_folder", default=None, type=str, required=True,
                        help="This is the working director, where we will find generation_file and \
                              where we will output the filtered out file")
    parser.add_argument("--generation_file", default=None, type=str, required=True,
                        help="The generation output script file")

    args = parser.parse_args()

    sentences = read_file(os.path.join(args.model_folder, args.generation_file))
    truncated = truncate(sentences)
    write_to_file(truncated, args.model_folder, args.generation_file, 'filtered_triggers_')

if __name__ == "__main__":
    main()
