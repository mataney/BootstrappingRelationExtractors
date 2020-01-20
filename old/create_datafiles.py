import argparse
import torch
import os
import json

from relation_canonical_form import CANONICAL_FORMS

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

CLEANINGMAP = {'-RRB-': ')', '-LRB-': '(', '-LSB-': '[',
               '-RSB-': ']', '-LCB-': '{', '-RCB-': '}',
               '&nbsp;': ' ', '&quot;': "'", '--': '-', '---': '-'}

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}
START_SUBJ = '<|subj|>'
END_SUBJ = '<|/subj|>'
START_OBJ = '<|obj|>'
END_OBJ = '<|/obj|>'
START_TRIGGER = '<|trigger|>'
END_TRIGGER = '<|/trigger|>'
GO = '<|GO|>'
ENCODER_AGNOSTIC_PAD = "SHALL" #For some reason SHALL is the padding token for encoder-agnostic

NO_RELATION = "no_relation"

RELATIONS_TO_LEAVE_OUT = []

def main(args):
    SPECIAL_TOKENS = [GO]
    if args.mark_relation_args:
        SPECIAL_TOKENS += [START_SUBJ, END_SUBJ, START_OBJ, END_OBJ, START_TRIGGER, END_TRIGGER]

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # Add Special Tokens
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})


    assert os.path.isfile(args.file_path)

    with open(args.file_path, encoding="utf-8") as f:
        parsed_json = json.load(f)

    srcs, tgts = [], []
    if args.agenda:
        agendas = []
    for relation_dict in parsed_json:
        if relation_dict['relation'] == NO_RELATION and not args.allow_no_relation:
            continue

        if leave_some_relations_out(relation_dict['relation']):
            continue

        subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx = [relation_dict[key] for key in ['subj_start', 'subj_end', 'obj_start', 'obj_end']]

        if args.anonymize:
            subj = relation_dict['subj_type']
            obj = relation_dict['obj_type']
            example_text = [relation_dict['token'][i] if ner == 'O' else ner for i, ner in enumerate(relation_dict['stanford_ner'])]
        else:
            subj = " ".join(relation_dict['token'][subj_start_idx : subj_end_idx + 1])
            obj = " ".join(relation_dict['token'][obj_start_idx : obj_end_idx + 1])
            example_text = relation_dict['token']

        if args.mark_relation_args:
            example_text = mark_args(example_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
        elif args.truncate_noise:
            example_text = truncate_noise(example_text, subj_start_idx, subj_end_idx + 1, obj_start_idx, obj_end_idx + 1)

        cleaned_example = clean_token(example_text)
        tgt = " ".join(cleaned_example)

        if args.one_form_per_relation:
            relation_contexts = [CANONICAL_FORMS[relation_dict['relation']][0]]
        else:
            relation_contexts = CANONICAL_FORMS[relation_dict['relation']]
        for relation_context in relation_contexts:
            if not args.first_four_as_src:
                src = relation_context.replace("{subj}", subj).replace("{obj}", obj)
            else:
                src = " ".join(cleaned_example[:4])
                if src == "":
                    continue
            if args.agenda:
                if args.only_one_token_subj_obj:
                    subj = one_token_per_arg(tokenizer, subj)
                    obj = one_token_per_arg(tokenizer, obj)
                    if subj is None or obj is None:
                        # Couldn't find subj or obj subwords that hav just one token
                        continue
                agendas.append(f"{subj} {ENCODER_AGNOSTIC_PAD} {obj}\n")
            srcs.append(src+'\n')
            tgts.append(tgt+'\n')
        
    with open(args.save_to_file+'.src', 'w') as f: f.writelines(srcs)
    with open(args.save_to_file+'.tgt', 'w') as f: f.writelines(tgts)
    if args.agenda:
        with open(args.save_to_file+'.agenda', 'w') as f: f.writelines(agendas)

def one_token_per_arg(tokenizer, word):
    tokens = word.split()
    for t in tokens:
        token_id = tokenizer.encode(t, add_prefix_space=True)
        if len(token_id) == 1:
            return t
    return None

def bio(tokens):
    insides = " ".join([f"{t}|I" for t in tokens[1:]])
    if insides:
        return f"{tokens[0]}|B {insides}"
    return f"{tokens[0]}|B"

def mark_args(text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx):
    if obj_end_idx > subj_end_idx:
        text.insert(obj_end_idx + 1, END_OBJ)
        text.insert(obj_start_idx, START_OBJ)
        text.insert(subj_end_idx + 1, END_SUBJ)
        text.insert(subj_start_idx, START_SUBJ)
    else:
        text.insert(subj_end_idx + 1, END_SUBJ)
        text.insert(subj_start_idx, START_SUBJ)
        text.insert(obj_end_idx + 1, END_OBJ)
        text.insert(obj_start_idx, START_OBJ)

    return text

def truncate_noise(example_text, subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx):
    padding = 0
    min_token_position = min(subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
    min_token_position = max(min_token_position - padding, 0)

    max_token_position = max(subj_start_idx, subj_end_idx, obj_start_idx, obj_end_idx)
    max_token_position = min(max_token_position + padding, len(example_text))

    return example_text[min_token_position:max_token_position]

def leave_some_relations_out(relation):
    return relation in RELATIONS_TO_LEAVE_OUT

def clean_token(tokens):
    return [CLEANINGMAP.get(t, t) for t in tokens]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default=None, type=str, required=True)
    parser.add_argument("--save_to_file", default=None, type=str, required=True)
    parser.add_argument("--anonymize", action='store_true')
    parser.add_argument("--mark_relation_args", action='store_true')
    parser.add_argument("--allow_no_relation", action='store_true')
    parser.add_argument("--truncate_noise", action='store_true')
    parser.add_argument("--one_form_per_relation", action='store_true')
    parser.add_argument("--first_four_as_src", action='store_true')
    parser.add_argument("--agenda", action='store_true')
    parser.add_argument("--only_one_token_subj_obj", action='store_true')

    args = parser.parse_args()
    args.model_type = 'gpt2'
    args.model_name_or_path = 'gpt2'
    args.config_name = ""
    args.tokenizer_name = ""
    args.do_lower_case = False
    args.block_size = 512
    args.local_rank = -1
    main(args)