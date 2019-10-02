import argparse
import os

from utils import read_file

def filter_out(sentences, triggers):
    filtered_sentences = []
    for sent in sentences:
        for trigger in triggers:
            if trigger in sent:
                filtered_sentences.append(sent)
                break

    return filtered_sentences

def write_to_file(filtered_sentences, model_folder, generation_file):
    output_file_path = os.path.join(model_folder, 'filtered_' + generation_file)
    with open(output_file_path, 'w') as file:
        for sent in filtered_sentences:
            file.write(f"{sent}\n")

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_folder", default=None, type=str, required=True,
                        help="This is the working director, where we will find generation_file and \
                              where we will output the filtered out file")
    parser.add_argument("--generation_file", default=None, type=str, required=True,
                        help="The generation output script file")
    parser.add_argument("--trigger_list_path", default=None, type=str, required=True,
                        help="Path of the list of triggers corresponding to a relation")

    args = parser.parse_args()

    sentences = read_file(os.path.join(args.model_folder, args.generation_file))
    triggers = read_file(args.trigger_list_path, remove_duplicates=True)
    filtered_sentences = filter_out(sentences, triggers)
    write_to_file(filtered_sentences, args.model_folder, args.generation_file)

if __name__ == "__main__":
    main()
