# pylint: disable=fixme,missing-function-docstring
# This script evaluates sentences according to their syntactic diversity.
# We clusters sentences according to their syntactic representation.
# Each cluster represents a differnt syntactic representation.
# The script uses SPIKE's anotation and ranking for clustering.

import spacy
from utils import read_file

#TODO: change this to be an argument
SENTENCES_PATH = '/Users/matane/matan/dev/datasets/junk/filtered_spouse_generation_outputs.txt'
TRIGGERS_LIST = '/Users/matane/matan/dev/datasets/trigger_lists/ \
                        SF_Resources/SF_Resources/trigger_dict_en/per_spouse_dict.xml'
ARG_NER = 'PERSON' #TODO this makes it specific to spouse

NLP = spacy.load("en_core_web_sm")


def read_sentences(path):
    with open(path) as f:
        sents = f.readlines()
    sents = [s.strip() for s in sents]
    return sents

def annotate_sentences(sentences):
    triggers_positions = _find_trigger(sentences)
    obj_sub_positions = _find_subj_obj(sentences)
    return ""

def _find_trigger(sentences):
    triggers_positions = []
    triggers = read_file(TRIGGERS_LIST, remove_duplicates=True)
    triggers.sort(key=len, reverse=True)
    for sent in sentences:
        tokenized_sent = NLP(sent)
        annotate_sent = [[t.text, t.i] for t in tokenized_sent if t.text in triggers]
        triggers_positions.append(annotate_sent[0]) #TODO Shuold deal with this, I assume there's only 1 trigger, this is not correct though.
    return triggers_positions

def _find_subj_obj(sentences):
    annotate_sentences = []
    for sent in sentences:
        tokenized_sent = NLP(sent)
        relevant_ents = [e for e in tokenized_sent.ents if e.label_ == ARG_NER] #TODO They should probably be different
        relevant_ents.sort(key=lambda e: e.end_char)
        if len(relevant_ents) == 2: #TODO dealing just with sents that have 2 entites.
            # I stopped here, after I found the "Relevant entities", should make them look like [s ent0]...[o ent1]
            # out_sent = sent[relevant_ents[0].end_char:]

            annotate_sent = " ".join([t.text if t.text not in relevant_ents else f"[t {t.text}]" for t in tokenized_sent])
            annotate_sentences.append(annotate_sent)
    return annotate_sentences

def do_something_with_spike(annotated_sentences):
    pass

def main():
    sentences = read_file(SENTENCES_PATH)
    annotated_sentences = annotate_sentences(sentences)
    do_something_with_spike(annotated_sentences)

if __name__ == '__main__':
    main()