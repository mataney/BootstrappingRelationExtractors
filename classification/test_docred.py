import json
import os

from classification.docred import DocREDUtils, DocREDProcessor

with open('classification/stubs/fake_truth.json', "r", encoding="utf-8") as f:
    docs = list(json.load(f))

doc1, doc2, doc3, doc4 = docs

# Tests using this variable require the true path to the data files.
DATA_DIR = '../datasets/DocRED/'

class TestDocREDUtils:
    def test__sents_entities_share(self):
        entities_sents = DocREDUtils._sents_entities_share(doc1, doc1['labels'][0])
        assert entities_sents == [0, 1]
        entities_sents = DocREDUtils._sents_entities_share(doc2, doc2['labels'][0])
        assert entities_sents == [0]
        entities_sents = DocREDUtils._sents_entities_share(doc3, doc3['labels'][0])
        assert entities_sents == [0, 1]
        entities_sents = DocREDUtils._sents_entities_share(doc3, doc3['labels'][1])
        assert entities_sents == [0, 1]

    def test__sents_entities_and_evidence_share(self):
        entities_sents = DocREDUtils._sents_entities_share(doc1, doc1['labels'][0])
        entities_and_evidence_sents = DocREDUtils._sents_entities_and_evidence_share(doc1['labels'][0], entities_sents)
        assert entities_and_evidence_sents == [1]
        entities_sents = DocREDUtils._sents_entities_share(doc2, doc2['labels'][0])
        entities_and_evidence_sents = DocREDUtils._sents_entities_and_evidence_share(doc2['labels'][0], entities_sents)
        assert entities_and_evidence_sents == [0]
        entities_sents = DocREDUtils._sents_entities_share(doc3, doc3['labels'][0])
        entities_and_evidence_sents = DocREDUtils._sents_entities_and_evidence_share(doc3['labels'][0], entities_sents)
        assert entities_and_evidence_sents == [0, 1]

    def test_entity_from_entity_id_passes(self):
        entity_list = DocREDUtils.entity_from_entity_id(doc1['vertexSet'], doc1['labels'][0]['h'], 0)
        assert entity_list == [{'name': 'Microsoft', 'pos': [0, 1], 'sent_id': 0, 'type': 'ORG'},
                               {'name': 'MS', 'pos': [3, 4], 'sent_id': 0, 'type': 'ORG'}]
        entity_list = DocREDUtils.entity_from_entity_id(doc1['vertexSet'], doc1['labels'][0]['h'], 1)
        assert entity_list[0] == {'name': 'Micro', 'pos': [0, 1], 'sent_id': 1, 'type': 'ORG'}

    def test_entities_by_sent_id(self):
        assert DocREDUtils.entities_by_sent_id(doc3['vertexSet']) == {0: {0, 1}, 1: {0, 1}}

    def test_relations_by_entities(self):
        assert DocREDUtils.relations_by_entities(doc3['labels']) == \
            {(0, 1): [{'r': 'P26', 'h': 0, 't': 1, 'evidence': [0, 1]}],
             (1, 0): [{'r': 'P26', 'h': 1, 't': 0, 'evidence': [0, 1]}]}

class TestDocREDProcessor:
    def test__same_entity_types_relation(self):
        processor = DocREDProcessor('founded_by')
        assert processor._same_entity_types_relation(doc1['labels'][0], doc1['vertexSet'])
        processor = DocREDProcessor('father')
        assert processor._same_entity_types_relation(doc2['labels'][0], doc2['vertexSet'])
        processor = DocREDProcessor('spouse')
        assert processor._same_entity_types_relation(doc3['labels'][0], doc3['vertexSet'])
        assert not processor._same_entity_types_relation(doc1['labels'][0], doc1['vertexSet'])

    def test__same_entity_types_relation_switched_h_and_t(self):
        processor = DocREDProcessor('founded_by')
        relation = doc1['labels'][0]
        head_is_tail = {'r': relation['r'], 'h': relation['t'], 't': relation['h'], 'evidence': relation['evidence']}
        assert not processor._same_entity_types_relation(head_is_tail, doc1['vertexSet'])

    def test__same_entity_types_relation_wrong_relation(self):
        processor = DocREDProcessor('inception')
        assert not processor._same_entity_types_relation(doc1['labels'][0], doc1['vertexSet'])

    def test_create_all_possible_dev_examples_doc1(self):
        processor = DocREDProcessor('founded_by')
        data = list(processor._create_all_possible_dev_examples([doc1], None))
        assert len(data) == 2
        assert data[0].evidence == 0
        assert data[0].h == 0
        assert data[0].t == 1
        assert data[0].label == 'NOTA'

        assert data[1].evidence == 1
        assert data[1].h == 0
        assert data[1].t == 1
        assert data[1].label == 'founded_by'

    def test_create_all_possible_dev_examples_doc2(self):
        processor = DocREDProcessor('father')
        data = list(processor._create_all_possible_dev_examples([doc2], None))
        assert len(data) == 2
        assert data[0].evidence == 0
        assert data[0].h == 0
        assert data[0].t == 1
        assert data[0].label == 'father'

        assert data[1].evidence == 0
        assert data[1].h == 1
        assert data[1].t == 0
        assert data[1].label == 'NOTA'

    def test_create_all_possible_dev_examples_doc3(self):
        processor = DocREDProcessor('spouse')
        data = list(processor._create_all_possible_dev_examples([doc3], None))
        assert len(data) == 4
        assert data[0].evidence == 0
        assert data[0].h == 0
        assert data[0].t == 1
        assert data[0].label == 'spouse'

        assert data[1].evidence == 0
        assert data[1].h == 1
        assert data[1].t == 0
        assert data[1].label == 'spouse'

        assert data[2].evidence == 1
        assert data[2].h == 0
        assert data[2].t == 1
        assert data[2].label == 'spouse'

        assert data[3].evidence == 1
        assert data[3].h == 1
        assert data[3].t == 0
        assert data[3].label == 'spouse'

    def test_create_all_possible_dev_examples_doc4(self):
        processor = DocREDProcessor('founded_by')
        data = list(processor._create_all_possible_dev_examples([doc4], None))
        assert len(data) == 1
        assert data[0].evidence == 0
        assert data[0].h == 0
        assert data[0].t == 1
        assert data[0].label == 'founded_by'

    def test_get_all_possible_eval_examples_check_positives(self):
        processor = DocREDProcessor('founded_by')
        data = processor.get_all_possible_eval_examples(DATA_DIR, 'dev')
        relations = [d for d in data if d.label == 'founded_by']
        distinct = list(set(relations))
        assert len(relations) == len(distinct)
