import csv
from collections import defaultdict
from itertools import permutations
import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, TypeVar, Set
from typing_extensions import TypedDict

from transformers.data.processors.utils import InputExample, InputFeatures
from classification.docred_config import RELATION_MAPPING, DOCRED_TACRED_RELATIONS_MAPPING, TACRED_DOCRED_RELATIONS_MAPPING
from classification.re_processors import REProcessor, JsonObject, wrap_text, SetType, NEGATIVE_LABEL

logger = logging.getLogger(__name__)

Relation = TypedDict('Relation', r=str, h=int, t=int, evidence=int)
Entity = TypedDict('Entity', name=str, pos=List[int], sent_id=int, type=str)
T = TypeVar('T', bound='DocREDExample')
Builder = Callable[[Type[T], int, JsonObject, Relation, str], List[T]]

class DocREDExample(InputExample):
    def __init__(self, id: int, text: str, label: str, evidence: int = 0, h: int = -1, t: int = -1) -> None:
        self.title = id
        self.evidence = evidence
        self.text = text
        self.label = label
        self.h = h
        self.t = t

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
    def build_annotated(cls: Type[T], title: int, example_json: JsonObject, relation: Relation, label: str = None) -> List[T]:
        for evidence in DocREDUtils.evidences_with_entities(example_json, relation):
            yield cls(id=title,
                      text=DocREDUtils.mark_entities(example_json, relation, evidence),
                      label=label,
                      evidence=evidence,
                      h=relation['h'],
                      t=relation['t'])

    @classmethod
    def build_distant(cls: Type[T], title: int, example_json: JsonObject, relation: Relation, label: str = None) -> List[T]:
        for evidence in DocREDUtils.sents_entities_share(example_json, relation):
            yield cls(id=title,
                      text=DocREDUtils.mark_entities(example_json, relation, evidence),
                      label=label,
                      evidence=evidence,
                      h=relation['h'],
                      t=relation['t'])

class DocREDUtils:
    @staticmethod
    def evidences_with_entities(example_json: JsonObject, relation: Relation) -> List[int]:
        entities_sents = DocREDUtils.sents_entities_share(example_json, relation)
        entities_and_evidence_sents = DocREDUtils._sents_entities_and_evidence_share(relation, entities_sents)
        return entities_and_evidence_sents

    @staticmethod
    def sents_entities_share(example_json: JsonObject, relation: Relation) -> List[int]:
        def sents_entity_appears_in(side: str) -> List[int]:
            return [e['sent_id'] for e in example_json['vertexSet'][relation[side]]]

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

    @staticmethod
    def mark_entities(example_json: JsonObject, relation: Relation, evidence: int) -> str:
        e1_start_idx, e1_end_idx = DocREDUtils._relation_span(example_json['vertexSet'], relation, 'h', evidence)
        e2_start_idx, e2_end_idx = DocREDUtils._relation_span(example_json['vertexSet'], relation, 't', evidence)
        text = example_json['sents'][evidence].copy()

        return wrap_text(text, e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx)

    @staticmethod
    def _relation_span(entities: List[Entity], relation: Relation, side: str, evidence: int) -> [int, int]:
        """
        Marking the first instance of the entity
        """
        entity = DocREDUtils.entity_from_entity_id(entities, relation[side], evidence)[0] # Assuming one wrapping will be enough
        return entity['pos'][0], entity['pos'][-1]

class DocREDProcessor(REProcessor):
    def __init__(self, relation_name: str, num_positive: int = None, negative_ratio: int = None, type_independent_neg_sample: bool = True) -> None:
        super().__init__(relation_name, num_positive, negative_ratio, type_independent_neg_sample)
        assert relation_name in RELATION_MAPPING
        self.relation_mapping = RELATION_MAPPING
        self.train_file = "train_split_from_annotated.json"
        self.dev_file = "eval_split_from_annotated.json"
        self.test_file = "dev.json"
        self.train_distant_file = "train_distant.json"

    def get_distant_train_examples(self, data_dir: str) -> List[DocREDExample]:
        """Gets a collection of `InputExample`s for the train set."""
        examples = self._create_examples(self._read_json(os.path.join(data_dir, self.train_distant_file)),
                                         "train_distant", builder=DocREDExample.build_distant)
        return self.sample_examples(examples, self.num_positive, self.negative_ratio)

    def _create_examples(self, documents: List[JsonObject],
                         set_type: SetType,
                         builder: Builder = DocREDExample.build_annotated) -> Iterator[DocREDExample]:
        """Creates examples for the training and dev sets."""
        for title_id, doc in enumerate(documents):
            for relation in doc['labels']:
                if self._positive_relation(relation) or self.allow_as_negative(relation, doc['vertexSet']):
                    examples = builder(title_id, doc, relation, label=self._relation_label(relation))
                    for example in examples:
                        yield example

    def _create_all_possible_dev_examples(self, documents: List[JsonObject], set_type: SetType) -> Iterator[DocREDExample]:
        """Creates examples of all possible entities for dev sets"""
        for title_id, doc in enumerate(documents):
            relations = self._create_all_relation_permutations(doc)
            for relation in relations:
                examples = DocREDExample.build_annotated(title_id, doc, relation, label=self._relation_label(relation))
                for example in examples:
                    yield example

    def _create_all_relation_permutations(self, doc: JsonObject) -> Iterator[Relation]:
        entities_by_sent_id = DocREDUtils.entities_by_sent_id(doc['vertexSet'])
        relations_by_entities = DocREDUtils.relations_by_entities(doc['labels'])

        relations = []

        positive_label_id = self.relation_mapping[self.positive_label]['id']
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

    def allow_as_negative(self, relation: Relation, entities: List[Entity]):
        return self.type_independent_neg_sample or self._same_entity_types_relation(relation, entities)

    def _same_entity_types_relation(self, relation: Relation, entities: List[Entity]) -> bool:
        """
        The try/catch essentially does the same the as in DocREDExample.validate.
        Don't need to use the validate method we don't need to log bad examples
        from possible negative examples.
        """
        def get_entity_type(side: str):
            return entities[relation[side]][0]['type']

        return get_entity_type('h') in self.relation_mapping[self.positive_label]['e1_type'] and \
               get_entity_type('t') in self.relation_mapping[self.positive_label]['e2_type']

    def _positive_relation(self, relation: Relation) -> bool:
        return relation['r'] == self.relation_mapping[self.positive_label]['id']

    def _positive_relation_name(self, relation_name: str) -> bool:
        return 1 if relation_name == self.positive_label else 0

    def _relation_label(self, relation: Relation) -> str:
        return 1 if self._positive_relation(relation) else 0

    def _create_search_examples_given_row_ids(self, search_file, row_ids: Set[int]) -> Iterator[InputExample]:
        with open(search_file, 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            return [DocREDExample(id=i,
                                  text=doc[0],
                                  label=self._positive_relation_name(self.reverse_relation_name_adapter(doc[1])))
                    for i, doc in enumerate(reader) if i in set(row_ids)]

    def _create_generation_examples(self, raw_generations: List[str]) -> Iterator[InputExample]:
        for i, gen in enumerate(raw_generations):
            yield DocREDExample(i, gen.rstrip(), 1)

    def relation_name_adapter(self, relation: str):
        return DOCRED_TACRED_RELATIONS_MAPPING[relation]

    def reverse_relation_name_adapter(self, relation: str):
        return TACRED_DOCRED_RELATIONS_MAPPING[relation]

class DocREDInputFeatures(InputFeatures):
    def __init__(self,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 markers_mask=None,
                 example=None,
                 label=None) -> None:
        super().__init__(input_ids, attention_mask, token_type_ids, label)
        self.markers_mask = markers_mask
        self.title = example.title
        self.h = example.h
        self.t = example.t
