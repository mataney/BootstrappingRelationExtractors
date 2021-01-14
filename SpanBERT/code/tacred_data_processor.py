import csv
import json
import logging
from math import ceil
import os
from random import shuffle, sample

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from classification.tacred_config import RELATION_MAPPING
from classification.re_config import RELATIONS_ENTITY_TYPES_FOR_SEARCH

CLS = "[CLS]"
SEP = "[SEP]"

class InputExample(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Processor for the TACRED data set."""
    def __init__(self, relation_name, num_positive, negative_ratio):
        self.positive_label = relation_name
        self.num_positive = num_positive
        self.negative_ratio = negative_ratio
        self.relation_mapping = RELATION_MAPPING
        self.min_examples_per_relation = 20

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        return data

    def get_train_examples(self, data_dir, training_method):
        if training_method == "train":
            return self.get_annotated_examples(data_dir)
        elif training_method == "search":
            return self.get_search_examples(data_dir)
        else:
            raise Exception("Wrong set_type name")

    def get_search_examples(self, data_dir):
        search_folder = 'search/single_trigger_search'
        return self.create_search_examples(data_dir, search_folder, self.num_positive, self.negative_ratio)

    def create_search_examples(self,
                               data_dir,
                               search_folder,
                               num_positive,
                               negative_ratio):
        positive_examples = self.sample_search_examples(os.path.join(data_dir, search_folder),
                                                        num_positive,
                                                        self.relation_name_adapter(self.positive_label))
        negative_examples = self.sample_search_examples(os.path.join(data_dir, search_folder),
                                                        len(positive_examples) * negative_ratio,
                                                        self.relations_entity_types_for_search(self.positive_label))
        return sample(positive_examples + negative_examples, len(positive_examples + negative_examples))

    def sample_search_examples(self, data_dir, num_to_sample, relation):
        def count_search_results(file: str, relation_name: str):
            return json.load(open(file, 'r', encoding="utf-8"))[relation_name]

        num_of_patterns = count_search_results(os.path.join(data_dir, 'file_lengths.json'), relation)
        indices = self._sample_indices(num_of_patterns, num_to_sample)
        examples = list(self._create_search_examples_given_row_ids(
            os.path.join(data_dir, relation), set(indices)
        ))
        return examples

    def _sample_indices(self, num_of_patterns, num_to_sample):
        samples_per_pattern = [max(self.min_examples_per_relation, ceil(num_to_sample / len(num_of_patterns))) for _ in num_of_patterns]
        return self._equal_samples_per_pattern(num_of_patterns, samples_per_pattern)

    def relation_name_adapter(self, relation: str):
        return relation

    def relations_entity_types_for_search(self, relation: str):
        relation = self.relation_name_adapter(relation)
        entity_types = RELATIONS_ENTITY_TYPES_FOR_SEARCH[relation]
        return f"{relation}-{entity_types}"

    def _create_search_examples_given_row_ids(self, search_file, row_ids):
        ner1, ner2 = RELATIONS_ENTITY_TYPES_FOR_SEARCH[self.positive_label].split(':')
        with open(search_file, 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for i, doc in enumerate(reader):
                if i in row_ids:
                    subj_start, subj_end, obj_start, obj_end, unwrapped_sent = self.unwrap(doc[0])
                    yield InputExample(guid=f"search_{doc[3]}",
                                       sentence=unwrapped_sent,
                                       span1=(subj_start, subj_end),
                                       span2=(obj_start, obj_end),
                                       ner1=ner1,
                                       ner2=ner2,
                                       label=self._relation_label(doc[1]))

    @classmethod
    def unwrap(cls, sentence):
        sent_list = sentence.split()
        subj_start = sent_list.index('[E1]')
        subj_end = sent_list.index('[/E1]')
        obj_start = sent_list.index('[E2]')
        obj_end = sent_list.index('[/E2]')
        if subj_start < obj_start:
            subj_end -= 2
            obj_start -= 2
            obj_end -= 4
        else:
            obj_end -= 2
            subj_start -= 2
            subj_end -= 4
        sent_list.remove('[E1]')
        sent_list.remove('[/E1]')
        sent_list.remove('[E2]')
        sent_list.remove('[/E2]')
        return subj_start, subj_end, obj_start, obj_end, sent_list
    
    def _relation_label(self, relation_name, negative_examples = "no_relation") -> str:
        return relation_name if self._positive_relation(relation_name) else negative_examples

    @classmethod
    def _equal_samples_per_pattern(cls, num_of_patterns, nums_to_sample):
        ret = []
        file_shift = 0
        for pattern_examples, num_to_sample in zip(num_of_patterns.values(), nums_to_sample):
            if num_to_sample > pattern_examples:
                num_to_sample = pattern_examples
            indices = sample(range(file_shift, file_shift + pattern_examples), num_to_sample)
            file_shift += pattern_examples
            ret += indices

        return ret
    
    def get_annotated_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")
        return self.sample_examples(examples, self.num_positive, self.negative_ratio)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self, data_dir, negative_label="no_relation"):
        """See base class."""
        return [negative_label, self.positive_label]
        # dataset = self._read_json(os.path.join(data_dir, "train.json"))
        # count = Counter()
        # for example in dataset:
        #     count[example['relation']] += 1
        # logger.info("%d labels" % len(count))
        # # Make sure the negative label is alwyas 0
        # labels = [negative_label]
        # for label, count in count.most_common():
        #     logger.info("%s: %.2f%%" % (label, count * 100.0 / len(dataset)))
        #     if label not in labels:
        #         labels.append(label)
        # return labels

    def _create_examples(self, dataset, set_type, negative_label='no_relation'):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            if self._positive_relation(example['relation']):
                label = example['relation']
            elif self._allow_as_negative(example):
                label = negative_label
            else:
                continue
            sentence = [convert_token(token) for token in example['token']]
            assert example['subj_start'] >= 0 and example['subj_start'] <= example['subj_end'] \
                and example['subj_end'] < len(sentence)
            assert example['obj_start'] >= 0 and example['obj_start'] <= example['obj_end'] \
                and example['obj_end'] < len(sentence)
            examples.append(InputExample(guid=example['id'],
                             sentence=sentence,
                             span1=(example['subj_start'], example['subj_end']),
                             span2=(example['obj_start'], example['obj_end']),
                             ner1=example['subj_type'],
                             ner2=example['obj_type'],
                             label=label))
        return examples

    def sample_examples(self, examples, num_positive, negative_ratio, eval = False):
        examples = list(examples)
        if not eval:
            shuffle(examples)
        positive_examples = self.get_first_num_examples(examples, True, num_positive)
        negative_examples = self.get_first_num_examples(examples, False, len(positive_examples) * negative_ratio)
        pos_and_neg_examples = positive_examples + negative_examples
        if not eval:
            shuffle(pos_and_neg_examples)

        return pos_and_neg_examples

    def get_first_num_examples(self, examples, positive, max_num):
        if positive:
            examples_in_label = [e for e in examples if e.label == self.positive_label]
        else:
            examples_in_label = [e for e in examples if e.label != self.positive_label]
        if max_num is None:
            return examples_in_label
        return examples_in_label[:max_num]

    def _positive_relation(self, relation_name: str) -> bool:
        return relation_name == self.positive_label

    def _allow_as_negative(self, example):
        return self._same_entity_types_relation(example)

    def _same_entity_types_relation(self, example):
        return (example['subj_type'] in self.relation_mapping[self.positive_label]['subj_type'] and
                example['obj_type'] in self.relation_mapping[self.positive_label]['obj_type'])


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, mode='text'):
    """Loads a data file into a list of `InputBatch`s."""


    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")
        SUBJECT_NER = get_special_token("SUBJ=%s" % example.ner1)
        OBJECT_NER = get_special_token("OBJ=%s" % example.ner2)

        if mode.startswith("text"):
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_START)
                if i == example.span2[0]:
                    tokens.append(OBJECT_START)
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                if i == example.span1[1]:
                    tokens.append(SUBJECT_END)
                if i == example.span2[1]:
                    tokens.append(OBJECT_END)
            if mode == "text_ner":
                tokens = tokens + [SEP, SUBJECT_NER, SEP, OBJECT_NER, SEP]
            else:
                tokens.append(SEP)
        else:
            subj_tokens = []
            obj_tokens = []
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_NER)
                if i == example.span2[0]:
                    tokens.append(OBJECT_NER)
                if (i >= example.span1[0]) and (i <= example.span1[1]):
                    for sub_token in tokenizer.tokenize(token):
                        subj_tokens.append(sub_token)
                elif (i >= example.span2[0]) and (i <= example.span2[1]):
                    for sub_token in tokenizer.tokenize(token):
                        obj_tokens.append(sub_token)
                else:
                    for sub_token in tokenizer.tokenize(token):
                        tokens.append(sub_token)
            if mode == "ner_text":
                tokens.append(SEP)
                for sub_token in subj_tokens:
                    tokens.append(sub_token)
                tokens.append(SEP)
                for sub_token in obj_tokens:
                    tokens.append(sub_token)
            tokens.append(SEP)
        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example.label]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token