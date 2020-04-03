import csv
from typing import Any, Callable, Dict, Iterator, List, Type, TypeVar
from typing_extensions import TypedDict

from classification.re_processors import REProcessor, JsonObject, wrap_text, NEGATIVE_LABEL, SetType
from classification.tacred_config import RELATION_MAPPING
from transformers.data.processors.utils import InputExample, InputFeatures

Relation = TypedDict('Relation', id=str, docid=str, relation=str, token=List[str], subj_start=int, subj_end=int, obj_start=int, obj_end=int, subj_type=str, obj_type=str, stanford_pos=List[str], stanford_ner=List[str], stanford_head=List[int], stanford_deprel=List[str])
T = TypeVar('T', bound='TACREDExample')
Builder = Callable[[Type[T], int, JsonObject, str], T]

class TACREDExample(InputExample):
    def __init__(self, id: int, example_json: JsonObject, label: str) -> None:
        self.id = id
        self.text = self._mark_entities(example_json)
        self.label = label

    def _mark_entities(self, example_json: JsonObject) -> str:
        e1_start_idx, e1_end_idx = example_json['subj_start'], example_json['subj_end']
        e2_start_idx, e2_end_idx = example_json['obj_start'], example_json['obj_end']
        text = example_json['token'].copy()

        return wrap_text(text, e1_start_idx, e1_end_idx + 1, e2_start_idx, e2_end_idx + 1)

    def __eq__(self, other: Any):
        if not isinstance(other, TACREDExample):
            return False

        if self.id == other.id and \
            self.text == other.text and \
            self.label == other.label:
            return True

        return False

    def __hash__(self):
        return hash((self.id, self.text, self.label))

    @classmethod
    def build(cls: Type[T], id: int, example_json: JsonObject, label: str) -> T:
        return cls(id, example_json, label)

class TACREDSearchExample(InputExample):
    def __init__(self, id: int, text: str, label: str) -> None:
        self.id = id
        self.text = text
        self.label = label

class TACREDProcessor(REProcessor):
    def __init__(self, relation_name: str, num_positive: int = None, negative_ratio: int = None, type_independent_neg_sample: bool = True) -> None:
        super().__init__(relation_name, num_positive, negative_ratio, type_independent_neg_sample)
        assert relation_name in RELATION_MAPPING
        self.relation_mapping = RELATION_MAPPING
        self.train_file = "train.json"
        self.dev_file = "dev.json"
        self.test_file = "test.json"

    def _create_examples(self, relations: Dict[int, Relation],
                         set_type: SetType,
                         builder: Builder = TACREDExample.build) -> Iterator[TACREDExample]:
        """Creates examples for the training and dev sets."""
        for id, relation in enumerate(relations):
            label = self._relation_label(relation['relation'])
            if self._positive_relation(label) or self.allow_as_negative(relation):
                yield builder(id, relation, label)

    def _create_all_possible_dev_examples(self,
                                          relations: Dict[int, Relation],
                                          set_type: SetType) -> Iterator[InputExample]:
        """Creates examples of all possible entities for dev sets"""
        for id, relation in enumerate(relations):
            label = self._relation_label(relation['relation'])
            if self._same_entity_types_relation(relation):
                yield TACREDExample.build(id, relation, label)

    def _create_search_examples_given_row_ids(self, search_file, row_ids: List[int]) -> Iterator[InputExample]:
        with open(search_file, 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for i, doc in enumerate(reader):
                if i in row_ids:
                    yield TACREDSearchExample(i, doc[0], doc[1])

    def relation_name_adapter(self, relation: str):
        return relation

    def _relation_label(self, relation_name: str) -> str:
        return 1 if self._positive_relation(relation_name) else 0

    def _positive_relation(self, relation_name: str) -> bool:
        return relation_name == self.positive_label

    def allow_as_negative(self, relation: Relation):
        return self.type_independent_neg_sample or self._same_entity_types_relation(relation)

    def _same_entity_types_relation(self, relation: Relation) -> bool:
        return (relation['subj_type'] in self.relation_mapping[self.positive_label]['subj_type'] and
                relation['obj_type'] in self.relation_mapping[self.positive_label]['obj_type'])

class TACREDInputFeatures(InputFeatures):
    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 markers_mask=None,
                 example=None,
                 label=None) -> None:
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.markers_mask = markers_mask
        self.title = example.id
        self.h = -1
        self.t = -1
