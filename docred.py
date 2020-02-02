import logging
import os
import json

from random import shuffle

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import acc_and_f1
from docred_config import (START_E1,
                           END_E1,
                           START_E2,
                           END_E2,
                           SPECIAL_TOKENS,
                           CLASS_MAPPING)

logger = logging.getLogger(__name__)

class DocREDExample(InputExample):

    def __init__(self, guid, example, relation, label=None):
        self.guid = guid
        self.text = self._mark_entities(example, relation)
        self.relation = relation
        self.label = label

    @classmethod
    def builder(cls, guid, example, relation, label=None):
        if cls.validate(example, relation):
            return cls(guid, example, relation, label)
        return None

    def _mark_entities(self, example, relation):
        e1_start_idx, e1_end_idx = self._relation_span(example, relation, 'h')
        e2_start_idx, e2_end_idx = self._relation_span(example, relation, 't')

        evidence = relation['evidence'][0]
        text = example['sents'][evidence].copy()

        if e2_end_idx > e1_end_idx:
            text.insert(e2_end_idx, END_E2)
            text.insert(e2_start_idx, START_E2)
            text.insert(e1_end_idx, END_E1)
            text.insert(e1_start_idx, START_E1)
        else:
            text.insert(e1_end_idx, END_E1)
            text.insert(e1_start_idx, START_E1)
            text.insert(e2_end_idx, END_E2)
            text.insert(e2_start_idx, START_E2)

        return ' '.join(text)

    def _relation_span(self, example, relation, side):
        """
        Marking the first instance of the entity
        """
        entity = DocREDUtils.entity_from_relation(example, relation, side)[0]
        return entity['pos'][0], entity['pos'][-1]

    @staticmethod
    def validate(example, relation):
        if not DocREDUtils.entitiy_found_in_sent(DocREDUtils.entity_from_relation(example, relation, 'h')):
            logger.info("Problem in annotation, can't find entity h in %s sent.", relation['evidence'][0])
            return False
        if not DocREDUtils.entitiy_found_in_sent(DocREDUtils.entity_from_relation(example, relation, 't')):
            logger.info("Problem in annotation, can't find entity t in %s sent.", relation['evidence'][0])
            return False
        if DocREDUtils.longer_than_one_sent(relation):
            logger.info("Skipping example: evidence is longer than 1 sentence.")
            return False
        return True

class DocREDUtils:
    @staticmethod
    def entity_from_relation(example, relation, side):
        return [e for e in example['vertexSet'][relation[side]] if e['sent_id'] == relation['evidence'][0]]

    @staticmethod
    def entitiy_found_in_sent(entity):
        return len(entity) != 0

    @staticmethod
    def longer_than_one_sent(relation):
        return len(relation['evidence']) > 1

class DocREDProcessor(DataProcessor):
    def __init__(self, relation_name, num_positive=None, num_negative=None):
        super().__init__()
        assert relation_name in CLASS_MAPPING
        self.positive_label = relation_name
        self.num_positive = num_positive
        self.num_negative = num_negative

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "train_annotated.json")), "train")
        return self.sample_examples(examples, self.num_positive, self.num_negative)

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")
        return self.sample_examples(examples, None, self.num_negative)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["1", "0"]

    @classmethod
    def _read_json(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(json.load(f))

    @staticmethod
    def sample_examples(examples, num_positive, num_negative):
        def get_first_num_examples(label, max_num):
            examples_in_label = [e for e in examples if e.label == label]
            if max_num is None:
                return examples_in_label
            return examples_in_label[:max_num]
        
        examples = list(examples)
        shuffle(examples)
        positive_examples = get_first_num_examples("1", num_positive)
        negative_examples = get_first_num_examples("0", num_negative)

        return positive_examples + negative_examples

    def _in_negative_relations(self, example, relation):
        """
        The try/catch essentially does the same the as in DocREDExample.validate.
        Don't need to use the validate method we don't need to log bad examples
        from possible negative examples.
        """
        try:
            entity_types = {CLASS_MAPPING[self.positive_label]['e1_type'],
                            CLASS_MAPPING[self.positive_label]['e2_type']}
            def same_entity_types():
                ent1_type = DocREDUtils.entity_from_relation(example, relation, 'h')[0]['type']
                ent2_type = DocREDUtils.entity_from_relation(example, relation, 't')[0]['type']
                
                return {ent1_type, ent2_type} == entity_types

            return same_entity_types()
        except IndexError as _:
            return False

    def _positive_relation(self, relation):
        return relation['r'] == self.positive_label

    def _relation_flag(self, relation):
        return "1" if self._positive_relation(relation) else "0"

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        i = 0
        for line in lines:
            for relation in line['labels']:
                guid = "%s-%s" % (set_type, i)
                if self._positive_relation(relation) or self._in_negative_relations(line, relation):
                    example = DocREDExample.builder(guid, line, relation, label=self._relation_flag(relation))
                    if example is not None:
                        yield example
                        i += 1

#This is a copy of glue_convert_examples_to_features with minor changes
def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    return features

def compute_metrics(task_name, preds, labels):
    assert task_name == "docred"
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

output_modes = {"docred": "classification"}

processors = {"docred": DocREDProcessor}
