import json
import logging
from math import ceil
import os
from random import shuffle
from typing import Any, Callable, Dict, Iterator, List
from typing_extensions import Literal
from random import sample

from sklearn.metrics import f1_score, precision_recall_fscore_support

from transformers.data.processors.utils import DataProcessor, InputExample
from classification.re_config import (START_E1,
                                      END_E1,
                                      START_E2,
                                      END_E2, 
                                      RELATIONS_ENTITY_TYPES_FOR_SEARCH)

SetType = Literal["train", "distant", "dev_eval", "full_dev_eval", "full_test_eval"]
logger = logging.getLogger(__name__)

JsonObject = Dict[str, Any]
Task = Literal["docred", "tacred"]
Builder = Callable[[List[Any]], List[InputExample]]

NEGATIVE_LABEL = "NOTA"

def wrap_text(text: List[str], e1_start_idx: int, e1_end_idx: int,
              e2_start_idx: int, e2_end_idx: int) -> str:
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

class REProcessor(DataProcessor):
    def __init__(self, relation_name: str, num_positive: int = None, negative_ratio: int = None, type_independent_neg_sample: bool = True) -> None:
        super().__init__()
        self.positive_label = relation_name
        self.num_positive = num_positive
        self.negative_ratio = negative_ratio
        self.type_independent_neg_sample = type_independent_neg_sample
        self.train_file, self.dev_file, self.test_file = None, None, None
        self.relation_mapping = None

    def get_examples_by_set_type(self, set_type: SetType, data_dir: str) -> List[InputExample]:
        if set_type == "train":
            return self.get_train_examples(data_dir)
        elif set_type == "distant":
            return self.get_distant_train_examples(data_dir)
        elif set_type == "search":
            return self.get_search_train_examples(data_dir)
        elif set_type == "dev_eval":
            return self.get_eval_examples(data_dir)
        elif set_type == "full_dev_eval":
            return self.get_all_possible_eval_examples(data_dir, 'full_dev_eval')
        elif set_type == "full_test_eval":
            return self.get_all_possible_eval_examples(data_dir, 'full_test_eval')
        else:
            raise Exception("Wrong set_type name")

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the train set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, self.train_file)), "train")
        return self.sample_examples(examples, self.num_positive, self.negative_ratio)

    def get_search_train_examples(self, data_dir: str) -> List[InputExample]:
        return self.create_search_examples(data_dir, self.num_positive, self.negative_ratio)

    def create_search_examples(self,
                               data_dir: str,
                               num_positive: int = None,
                               negative_ratio: int = None) -> List[InputExample]:
        search_folder = 'search/single_trigger_search'
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

        pattern_counts = count_search_results(os.path.join(data_dir, 'file_lengths.json'), relation)
        indices = self._equal_samples_per_pattern(pattern_counts, num_to_sample)
        examples = self._create_search_examples_given_row_ids(
            os.path.join(data_dir, relation), indices
        )
        return list(examples)

    def _create_search_examples_given_row_ids(self, search_file, row_ids: List[int]) -> Iterator[InputExample]:
        raise NotImplementedError

    def relations_entity_types_for_search(self, relation: str):
        relation = self.relation_name_adapter(relation)
        entity_types = RELATIONS_ENTITY_TYPES_FOR_SEARCH[relation]
        return f"{relation}-{entity_types}"

    def relation_name_adapter(self, relation: str):
        raise NotImplementedError

    def get_eval_examples(self, data_dir: str) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the dev set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, self.dev_file)), "dev")
        return self.sample_examples(examples, self.num_positive, self.negative_ratio, eval=True)

    def get_all_possible_eval_examples(self, data_dir: str, set_type: SetType) -> List[InputExample]:
        """Gets a collection of `InputExample`s for the eval set."""
        if set_type == 'full_dev_eval':
            examples = self._create_all_possible_dev_examples(self._read_json(os.path.join(data_dir, self.dev_file)), "full_dev_eval")
        elif set_type == 'full_test_eval':
            examples = self._create_all_possible_dev_examples(self._read_json(os.path.join(data_dir, self.test_file)), "full_test_eval")
        else:
            raise Exception("Wrong set_type name")
        return list(examples)

    def sample_examples(self, examples: List[JsonObject],
                        num_positive: int = None,
                        negative_ratio: int = None,
                        eval: bool = False) -> List[InputExample]:
        def get_first_num_examples(label, max_num):
            examples_in_label = [e for e in examples if e.label == label]
            if max_num is None:
                return examples_in_label
            return examples_in_label[:max_num]

        examples = list(examples)
        if not eval:
            shuffle(examples)
        positive_examples = get_first_num_examples(self.positive_label, num_positive)
        negative_examples = get_first_num_examples(NEGATIVE_LABEL, len(positive_examples) * negative_ratio)
        pos_and_neg_examples = positive_examples + negative_examples
        if not eval:
            shuffle(pos_and_neg_examples)

        return pos_and_neg_examples

    def _create_examples(self, documents: List[JsonObject],
                         set_type: Any,
                         builder: Builder = None) -> Iterator[InputExample]:
        raise NotImplementedError

    def _create_search_examples(self, documents: List[str]) -> List[InputExample]:
        raise NotImplementedError

    def _create_all_possible_dev_examples(self,
                                          documents: List[JsonObject],
                                          set_type: SetType) -> Iterator[InputExample]:
        raise NotImplementedError

    def get_distant_train_examples(self, data_dir: str) -> List[InputExample]:
        raise NotImplementedError

    @classmethod
    def _read_json(cls, input_file: str) -> List[JsonObject]:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(json.load(f))

    @classmethod
    def _equal_samples_per_pattern(cls, pattern_counts, num_to_sample):
        ret = []
        num_to_sample = ceil(num_to_sample / len(pattern_counts))
        file_shift = 0
        for _, pattern_examples in pattern_counts.items():
            if num_to_sample > pattern_examples:
                num_to_sample = pattern_examples
            indices = sample(range(file_shift, file_shift + pattern_examples), num_to_sample)
            file_shift += pattern_examples
            ret += indices

        return ret

    def get_labels(self) -> List[str]:
        """Gets the list of labels for this data set."""
        return [self.positive_label, NEGATIVE_LABEL]

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
    input_features_class = input_features_factory(task)
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
            logger.info("example: %s" % (example))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("markers_mask: %s" % " ".join([str(x) for x in markers_mask]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            input_features_class(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                markers_mask=markers_mask,
                example=example,
                label=label,
            )
        )

    return features


def input_features_factory(task: Task):
    if task == 'docred':
        from classification.docred import DocREDInputFeatures
        return DocREDInputFeatures
    elif task == 'tacred':
        from classification.tacred import TACREDInputFeatures
        return TACREDInputFeatures
    else:
        raise ValueError("Task not found: %s" % (task))

def compute_metrics(task_name: Task, preds, labels, positive_label):
    assert task_name == "docred" or task_name == "tacred"
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

class Processors:
    def __getitem__(self, task: Task):
        if task == 'docred':
            from classification.docred import DocREDProcessor
            return DocREDProcessor
        elif task == 'tacred':
            from classification.tacred import TACREDProcessor
            return TACREDProcessor
        else:
            raise ValueError("Task not found: %s" % (task))

class RelationMapping:
    def __getitem__(self, task: Task):
        if task == 'docred':
            from classification.docred_config import RELATION_MAPPING
            return RELATION_MAPPING
        elif task == 'tacred':
            from classification.tacred_config import RELATION_MAPPING
            return RELATION_MAPPING
        else:
            raise ValueError("Task not found: %s" % (task))

class TitleNames:
    def __call__(self, task: Task, set_type: SetType, t: int):
        if task == "docred":
            from classification.docred_config import DEV_TITLES, TRAIN_EVAL_TITLES
            if set_type == 'full_dev_eval':
                return TRAIN_EVAL_TITLES[t]
            elif set_type == 'full_test_eval':
                return DEV_TITLES[t]
            else:
                raise ValueError("Set type not found: %s" % (set_type))
        elif task == 'tacred':
            return int(t) # The name is the ID
        else:
            raise ValueError("Task not found: %s" % (task))

output_modes = {"docred": "classification", "tacred": "classification"}
processors = Processors()
RELATION_MAPPING = RelationMapping()
TITLE_NAMES = TitleNames()