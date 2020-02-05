import json
import logging
import os
from typing import List, Dict, Any, Iterator, Tuple, Type, TypeVar
from typing_extensions import Literal, TypedDict

from collections import defaultdict
from itertools import permutations
from random import shuffle

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import acc_and_f1
from docred_config import (START_E1,
                           END_E1,
                           START_E2,
                           END_E2,
                           CLASS_MAPPING)

logger = logging.getLogger(__name__)

JsonObject = Dict[str, Any]
SetType = Literal["train", "dev", "full_dev"]
Relation = TypedDict('Relation', r=str, h=int, t=int, evidence=List[int])
Entity = TypedDict('Entity', name=str, pos=List[int], sent_id=int, type=str)
T = TypeVar('T', bound='DocREDExample')

class DocREDExample(InputExample):

    def __init__(self, title: int, example_json: JsonObject, relation: Relation, label: str = None) -> None:
        self.title = title
        self.text = self._mark_entities(example_json, relation)
        self.h = relation['h']
        self.t = relation['t']
        self.label = label

    @classmethod
    def build(cls: Type[T], title: int, example_json: JsonObject, relation: Relation, label=None) -> T:
        if cls.validate(example_json, relation):
            return cls(title, example_json, relation, label)
        return None

    def _mark_entities(self, example_json: JsonObject, relation: Relation) -> str:
        e1_start_idx, e1_end_idx = self._relation_span(example_json, relation, 'h')
        e2_start_idx, e2_end_idx = self._relation_span(example_json, relation, 't')

        evidence = relation['evidence'][0]
        text = example_json['sents'][evidence].copy()

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

    def _relation_span(self, example_json: JsonObject, relation: Relation, side: str) -> [int, int]:
        """
        Marking the first instance of the entity
        """
        entity = DocREDUtils.entity_from_relation(example_json['vertexSet'], relation, side)[0]
        return entity['pos'][0], entity['pos'][-1]

    @staticmethod
    def validate(example_json: JsonObject, relation: Relation) -> bool:
        if not DocREDUtils.entitiy_found_in_sent(DocREDUtils.entity_from_relation(example_json['vertexSet'], relation, 'h')):
            logger.info("Problem in annotation, can't find entity h in %s sent.", relation['evidence'][0])
            return False
        if not DocREDUtils.entitiy_found_in_sent(DocREDUtils.entity_from_relation(example_json['vertexSet'], relation, 't')):
            logger.info("Problem in annotation, can't find entity t in %s sent.", relation['evidence'][0])
            return False
        if DocREDUtils.longer_than_one_sent(relation):
            logger.info("Skipping example: evidence is longer than 1 sentence.")
            return False
        return True

class DocREDUtils:
    @staticmethod
    def entity_from_relation(entites: List[Entity], relation: Relation, side: str) -> List[Entity]:
        return [e for e in entites[relation[side]] if e['sent_id'] == relation['evidence'][0]]

    @staticmethod
    def entitiy_found_in_sent(entity) -> bool:
        return len(entity) != 0

    @staticmethod
    def longer_than_one_sent(relation: Relation) -> bool:
        return len(relation['evidence']) > 1

    @staticmethod
    def entities_by_sent_id(entities: List[Entity]) -> Dict[int, List[int]]:
        grouped = defaultdict(set)
        for i, ent_instances in enumerate(entities):
            for ent in ent_instances:
                grouped[ent['sent_id']].add(i)
        return grouped

    @staticmethod
    def relations_by_entities(relations: List[Relation]) -> Dict[Tuple[int, int], str]:
        grouped = defaultdict(list)
        for relation in relations:
            grouped[relation['h'], relation['t']] = relation
        return grouped

class DocREDProcessor(DataProcessor):
    def __init__(self, relation_name: str, num_positive: int = None, num_negative: int = None) -> None:
        super().__init__()
        assert relation_name in CLASS_MAPPING
        self.positive_label = relation_name
        self.num_positive = num_positive
        self.num_negative = num_negative

    def get_examples_by_set_type(self, set_type: SetType, data_dir: str) -> List[DocREDExample]:
        if set_type == "train":
            return self.get_train_examples(data_dir)
        elif set_type == "dev":
            return self.get_dev_examples(data_dir)
        elif set_type == "full_dev":
            return self.get_all_possible_dev_examples(data_dir)
        else:
            raise Exception("Wrong set_type name")

    def get_train_examples(self, data_dir: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the train set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "train_annotated.json")), "train")
        return self.sample_examples(examples, self.num_positive, self.num_negative)

    def get_dev_examples(self, data_dir: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")
        return self.sample_examples(examples, self.num_negative, self.num_negative, eval=True)

    def get_all_possible_dev_examples(self, data_dir: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        examples = self._create_all_possible_dev_examples(self._read_json(os.path.join(data_dir, "dev.json")), "full_dev")
        return list(examples)

    def _create_examples(self, documents: List[JsonObject], set_type: SetType) -> Iterator[DocREDExample]:
        """Creates examples for the training and dev sets."""
        for title_id, doc in enumerate(documents):
            for relation in doc['labels']:
                if self._positive_relation(relation) or self._same_entity_types_relation(doc, relation):
                    example = DocREDExample.build(title_id, doc, relation, label=self._relation_label(relation))
                    if example is not None:
                        yield example
    
    def _create_all_possible_dev_examples(self, documents: List[JsonObject], set_type: SetType) -> Iterator[DocREDExample]:
        """Creates examples of all possible entities for dev sets"""
        for title_id, doc in enumerate(documents):
            relations = self._create_all_relation_permutations(doc)
            for relation in relations:
                example = DocREDExample.build(title_id, doc, relation, label=self._relation_label(relation))
                if example is not None:
                    yield example

    def _create_all_relation_permutations(self, doc: JsonObject) -> Iterator[Relation]:
        entities_by_sent_id = DocREDUtils.entities_by_sent_id(doc['vertexSet'])
        relations_by_entities = DocREDUtils.relations_by_entities(doc['labels'])
        for line, entities in entities_by_sent_id.items():
            for perm in permutations(entities, 2):
                relation_name = relations_by_entities[perm]['r'] if perm in relations_by_entities else 'NOTA'
                relation_evidence = relations_by_entities[perm]['evidence'] if perm in relations_by_entities else [line]
                relation = {'r': relation_name, 'h': perm[0], 't': perm[1], 'evidence': relation_evidence}
                if self._same_entity_types_relation(doc, relation):
                    yield relation

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        return [self.positive_label, "NOTA"]

    @classmethod
    def _read_json(cls, input_file: str) -> List[JsonObject]:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(json.load(f))

    def sample_examples(self, examples: List[JsonObject], num_positive: int = None, num_negative: int = None, eval: bool = False) -> List[DocREDExample]:
        def get_first_num_examples(label, max_num):
            examples_in_label = [e for e in examples if e.label == label]
            if max_num is None:
                return examples_in_label
            return examples_in_label[:max_num]

        examples = list(examples)
        if not eval:
            shuffle(examples)
        positive_examples = get_first_num_examples(self.positive_label, num_positive)
        negative_examples = get_first_num_examples("NOTA", num_negative)
        pos_and_neg_examples = positive_examples + negative_examples
        if not eval:
            shuffle(pos_and_neg_examples)

        return pos_and_neg_examples

    def _same_entity_types_relation(self, example_json: JsonObject, relation: Relation) -> bool:
        """
        The try/catch essentially does the same the as in DocREDExample.validate.
        Don't need to use the validate method we don't need to log bad examples
        from possible negative examples.
        """
        def same_entity_types():
            if not DocREDExample.validate(example_json, relation):
                return False

            ent1 = DocREDUtils.entity_from_relation(example_json['vertexSet'], relation, 'h')
            ent2 = DocREDUtils.entity_from_relation(example_json['vertexSet'], relation, 't')

            return {ent1[0]['type'], ent2[0]['type']} == entity_types

        try:
            entity_types = {CLASS_MAPPING[self.positive_label]['e1_type'],
                            CLASS_MAPPING[self.positive_label]['e2_type']}

            return same_entity_types()
        except IndexError as _:
            return False

    def _positive_relation(self, relation: Relation) -> bool:
        return relation['r'] == CLASS_MAPPING[self.positive_label]['id']

    def _relation_label(self, relation: Relation) -> str:
        return self.positive_label if self._positive_relation(relation) else "NOTA"

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

        start_markers_ids = [tokenizer.convert_tokens_to_ids(t) for t in [START_E1, START_E2]]
        markers_mask = [1 if t in start_markers_ids else 0 for t in input_ids]
        if sum(markers_mask) != 2:
            logger.info("Text is truncated and not all entities have made it.")
            continue

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
            logger.info("title: %s" % (example.title))
            logger.info("head: %s" % (example.h))
            logger.info("tail: %s" % (example.t))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("markers_mask: %s" % " ".join([str(x) for x in markers_mask]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            DocREDInputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                markers_mask=markers_mask,
                title=example.title,
                h=example.h,
                t=example.t,
                label=label,
            )
        )

    return features

class DocREDInputFeatures(InputFeatures):
    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 markers_mask=None,
                 title=None,
                 h=None,
                 t=None,
                 label=None) -> None:
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.markers_mask = markers_mask
        self.title = title
        self.h = h
        self.t = t

def compute_metrics(task_name, preds, labels):
    assert task_name == "docred"
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

output_modes = {"docred": "classification"}

processors = {"docred": DocREDProcessor}
