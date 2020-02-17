from collections import defaultdict
import json
from itertools import permutations
import logging
import os
from random import shuffle
from typing import List, Dict, Any, Iterator, Tuple, Type, TypeVar
from typing_extensions import Literal, TypedDict

from sklearn.metrics import f1_score, precision_recall_fscore_support

from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from classification.docred_config import (START_E1,
                                          END_E1,
                                          START_E2,
                                          END_E2,
                                          CLASS_MAPPING)

logger = logging.getLogger(__name__)

JsonObject = Dict[str, Any]
SetType = Literal["train", "dev", "full_dev"]
Relation = TypedDict('Relation', r=str, h=int, t=int, evidence=int)
Entity = TypedDict('Entity', name=str, pos=List[int], sent_id=int, type=str)
T = TypeVar('T', bound='DocREDExample')

NEGATIVE_LABEL = "NOTA"

class DocREDExample(InputExample):

    def __init__(self, title: int, example_json: JsonObject, relation: Relation, evidence: int, label: str = None) -> None:
        self.title = title
        self.evidence = evidence
        self.text = self._mark_entities(example_json, relation)
        self.h = relation['h']
        self.t = relation['t']
        self.label = label

    def __eq__(self, other: Any):
        if not isinstance(other, DocREDExample):
            return False

        if self.title == other.title and \
            self.text == other.text and \
            self.h == other.h and \
            self.t == other.t and \
            self.label == other.label:
            return True

        return False

    def __hash__(self):
        return hash((self.title, self.text, self.h, self.t, self.label))

    @classmethod
    def build(cls: Type[T], title: int, example_json: JsonObject, relation: Relation, label=None) -> List[T]:
        for evidence in DocREDUtils.evidences_with_entities(example_json, relation):
            yield cls(title, example_json, relation, evidence, label)

    def _mark_entities(self, example_json: JsonObject, relation: Relation) -> str:
        e1_start_idx, e1_end_idx = self._relation_span(example_json['vertexSet'], relation, 'h')
        e2_start_idx, e2_end_idx = self._relation_span(example_json['vertexSet'], relation, 't')

        text = example_json['sents'][self.evidence].copy()

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

    def _relation_span(self, entities: List[Entity], relation: Relation, side: str) -> [int, int]:
        """
        Marking the first instance of the entity
        """
        entity = DocREDUtils.entity_from_entity_id(entities, relation[side], self.evidence)[0] # Assuming one wrapping will be enough
        return entity['pos'][0], entity['pos'][-1]

class DocREDUtils:
    @staticmethod
    def evidences_with_entities(json_example: JsonObject, relation: Relation) -> List[int]:
        entities_sents = DocREDUtils._sents_entities_share(json_example, relation)
        entities_and_evidence_sents = DocREDUtils._sents_entities_and_evidence_share(relation, entities_sents)
        return entities_and_evidence_sents

    @staticmethod
    def _sents_entities_share(json_example: JsonObject, relation: Relation) -> List[int]:
        def sents_entity_appears_in(side: str) -> List[int]:
            return [e['sent_id'] for e in json_example['vertexSet'][relation[side]]]

        head_sents = sents_entity_appears_in('h')
        tail_sents = sents_entity_appears_in('t')

        return list(set(head_sents) & set(tail_sents))

    @staticmethod
    def _sents_entities_and_evidence_share(relation: Relation, entities_sents: List[int]) -> List[int]:
        return list(set(relation['evidence']) & set(entities_sents))

    @staticmethod
    def entity_from_entity_id(entities: List[Entity], entity_id: int, evidence: int) -> List[Entity]:
        return [e for e in entities[entity_id] if e['sent_id'] == evidence]

    @staticmethod
    def entities_by_sent_id(entities: List[Entity]) -> Dict[int, List[int]]:
        grouped = defaultdict(set)
        for i, ent_instances in enumerate(entities):
            for ent in ent_instances:
                grouped[ent['sent_id']].add(i)
        return grouped

    @staticmethod
    def relations_by_entities(relations: List[Relation]) -> Dict[Tuple[int, int], Relation]:
        grouped = defaultdict(list)
        for relation in relations:
            grouped[relation['h'], relation['t']].append(relation)
        return grouped

    @staticmethod
    def entities_in_positive_relation_in_this_sent(entities_ids: Tuple[int, int],
                                                    positive_label_id: str,
                                                    sent_id: int,
                                                    relations_by_entities: Dict[Tuple[int, int], Relation]) -> bool:

        if entities_ids not in relations_by_entities:
            return False

        for rel in relations_by_entities[entities_ids]:
            if positive_label_id == rel['r'] and sent_id in rel['evidence']:
                return True
        return False

class DocREDProcessor(DataProcessor):
    def __init__(self, relation_name: str, num_positive: int = None, num_negative: int = None, type_independent_neg_sample: bool = True) -> None:
        super().__init__()
        assert relation_name in CLASS_MAPPING
        self.positive_label = relation_name
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.type_independent_neg_sample = type_independent_neg_sample

    def get_examples_by_set_type(self, set_type: SetType, data_dir: str) -> List[DocREDExample]:
        if set_type == "train":
            return self.get_train_examples(data_dir)
        elif set_type == "train_eval":
            return self.get_eval_examples(data_dir)
        elif set_type == "full_train_eval":
            return self.get_all_possible_eval_examples(data_dir, 'full_train_eval')
        elif set_type == "full_dev_eval":
            return self.get_all_possible_eval_examples(data_dir, 'dev')
        else:
            raise Exception("Wrong set_type name")

    def get_train_examples(self, data_dir: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the train set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "train_split_from_annotated.json")), "train")
        return self.sample_examples(examples, self.num_positive, self.num_negative)

    def get_eval_examples(self, data_dir: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, "eval_split_from_annotated.json")), "dev")
        return self.sample_examples(examples, self.num_positive, self.num_negative, eval=True)

    def get_all_possible_eval_examples(self, data_dir: str, set_type: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the eval set."""
        if set_type == 'full_train_eval':
            examples = self._create_all_possible_dev_examples(self._read_json(os.path.join(data_dir, "eval_split_from_annotated.json")), "full_eval_split_from_annotated_eval")
        else:
            examples = self._create_all_possible_dev_examples(self._read_json(os.path.join(data_dir, "dev.json")), "full_dev_eval")
        return list(examples)

    def _create_examples(self, documents: List[JsonObject], set_type: SetType) -> Iterator[DocREDExample]:
        """Creates examples for the training and dev sets."""
        for title_id, doc in enumerate(documents):
            for relation in doc['labels']:
                if self._positive_relation(relation) or self.allow_as_negative(relation, doc['vertexSet']):
                    examples = DocREDExample.build(title_id, doc, relation, label=self._relation_label(relation))
                    for example in examples:
                        yield example

    def _create_all_possible_dev_examples(self, documents: List[JsonObject], set_type: SetType) -> Iterator[DocREDExample]:
        """Creates examples of all possible entities for dev sets"""
        for title_id, doc in enumerate(documents):
            relations = self._create_all_relation_permutations(doc)
            for relation in relations:
                examples = DocREDExample.build(title_id, doc, relation, label=self._relation_label(relation))
                for example in examples:
                    yield example

    def _create_all_relation_permutations(self, doc: JsonObject) -> Iterator[Relation]:
        entities_by_sent_id = DocREDUtils.entities_by_sent_id(doc['vertexSet'])
        relations_by_entities = DocREDUtils.relations_by_entities(doc['labels'])
        
        relations = []

        positive_label_id = CLASS_MAPPING[self.positive_label]['id']
        for sent_id, entities_in_sent in entities_by_sent_id.items():
            for perm in permutations(entities_in_sent, 2):
                relation_id = (
                    positive_label_id if DocREDUtils.entities_in_positive_relation_in_this_sent(perm,
                                                                                                positive_label_id,
                                                                                                sent_id,
                                                                                                relations_by_entities)
                    else NEGATIVE_LABEL
                )
                relations.append({'r': relation_id, 'h': perm[0], 't': perm[1], 'evidence': [sent_id]})


        for relation in relations:
            if self._same_entity_types_relation(relation, doc['vertexSet']):
                yield relation

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        return [self.positive_label, NEGATIVE_LABEL]

    def allow_as_negative(self, relation: Relation, entities: List[Entity]):
        return self.type_independent_neg_sample or self._same_entity_types_relation(relation, entities)

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
        negative_examples = get_first_num_examples(NEGATIVE_LABEL, num_negative)
        pos_and_neg_examples = positive_examples + negative_examples
        if not eval:
            shuffle(pos_and_neg_examples)

        return pos_and_neg_examples

    def _same_entity_types_relation(self, relation: Relation, entities: List[Entity]) -> bool:
        """
        The try/catch essentially does the same the as in DocREDExample.validate.
        Don't need to use the validate method we don't need to log bad examples
        from possible negative examples.
        """
        def get_entity_type(side: str):
            return entities[relation[side]][0]['type']

        entity_types = [CLASS_MAPPING[self.positive_label]['e1_type'],
                        CLASS_MAPPING[self.positive_label]['e2_type']]

        return [get_entity_type('h'), get_entity_type('t')] == entity_types

    def _positive_relation(self, relation: Relation) -> bool:
        return relation['r'] == CLASS_MAPPING[self.positive_label]['id']

    def _relation_label(self, relation: Relation) -> str:
        return self.positive_label if self._positive_relation(relation) else NEGATIVE_LABEL

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

def compute_metrics(task_name, preds, labels, positive_label):
    assert task_name == "docred"
    assert len(preds) == len(labels)
    return f1_ignore_negative_class(preds, labels, positive_label)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def f1_ignore_negative_class(preds, labels, positive_label):
    f1 = f1_score(y_true=labels, y_pred=preds)
    p, r, f, _ = precision_recall_fscore_support(y_true=labels,
                                                            y_pred=preds,
                                                            pos_label=positive_label,
                                                            average='binary')

    return {
        "p": p,
        "r": r,
        "f1": f,
        "f1_do_not_ignore_negative_class": f1,
    }

output_modes = {"docred": "classification"}

processors = {"docred": DocREDProcessor}
