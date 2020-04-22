import argparse
import os
from tqdm import tqdm

from old.utils import read_file, write_to_file
from spike.annotators.annotator_service import AnnotatorService

#Will probably need to add spike to the pythonpath
# and `source activate spike`

def filter_out(sentences, entities):
    annotator = AnnotatorService.from_env()
    filtered_sentences = []
    for sent in tqdm(sentences):
        annotated = annotator.annotate_text(sent)
        featuring_entities = [e.label.lower() for e in annotated.sentences[0].entities]
        found_required_entities = True
        for e in entities:
            e = e.lower()
            if e in featuring_entities:
                featuring_entities.remove(e)
            else:
                found_required_entities = False
        if found_required_entities:
            filtered_sentences.append(sent)

    return filtered_sentences

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_folder", default=None, type=str, required=True,
                        help="This is the working director, where we will find generation_file and \
                              where we will output the filtered out file")
    parser.add_argument("--generation_file", default=None, type=str, required=True,
                        help="The generation output script file")
    parser.add_argument('--entities', nargs='+', type=str, required=True,
                        help="The entities to look for")

    args = parser.parse_args()

    assert len(args.entities) == 2

    sentences = read_file(os.path.join(args.model_folder, args.generation_file))
    filtered_sentences = filter_out(sentences, args.entities)
    write_to_file(filtered_sentences, args.model_folder, args.generation_file, 'filtered_ents_')

if __name__ == "__main__":
    main()
