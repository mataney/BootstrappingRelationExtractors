from docred import DocREDExample, DocREDUtils

doc = {
    'vertexSet':
        [[{'name': 'Lark Force', 'pos': [0, 2], 'sent_id': 0, 'type': 'ORG'}], [{'name': 'Australian Army', 'pos': [4, 6], 'sent_id': 0, 'type': 'ORG'}], [{'pos': [9, 11], 'type': 'TIME', 'sent_id': 0, 'name': 'March 1941'}], [{'name': 'World War II', 'pos': [12, 15], 'sent_id': 0, 'type': 'MISC'}], [{'name': 'New Britain', 'pos': [18, 20], 'sent_id': 0, 'type': 'LOC'}], [{'name': 'New Ireland', 'pos': [21, 23], 'sent_id': 0, 'type': 'LOC'}]], 
    'labels':
        [{'r': 'P607', 'h': 1, 't': 3, 'evidence': [0]},
         {'r': 'P571', 'h': 0, 't': 2, 'evidence': [0]},
         {'r': 'P607', 'h': 0, 't': 3, 'evidence': [0]}],
    'title': 'Lark Force',
    'sents':
        [['Lark', 'Force', 'was', 'an', 'Australian', 'Army', 'formation', 'established', 'in', 'March', '1941', 'during', 'World', 'War', 'II', 'for', 'service', 'in', 'New', 'Britain', 'and', 'New', 'Ireland', '.']]
    }

entities_by_sent_id = {0: [0, 1, 2, 3, 4, 5]}
relations_by_entities = {(1, 3): 'P607', (0, 2): 'P571', (0, 3): 'P607'}

class TestDocREDExample:
    def test_validate_returns_true(self):
        assert DocREDExample.validate(doc, doc['labels'][0])

class TestDocREDUtils:
    def test_entity_from_relation_passes(self):
        entity_list = DocREDUtils.entity_from_relation(doc['vertexSet'], doc['labels'][0], 'h')
        assert entity_list[0] == {'name': 'Australian Army', 'pos': [4, 6], 'sent_id': 0, 'type': 'ORG'}

    def test_found_in_sent(self):
        assert DocREDUtils.entitiy_found_in_sent(DocREDUtils.entity_from_relation(doc['vertexSet'], doc['labels'][0], 'h'))

    def test_entities_by_sent_id(self):
        assert DocREDUtils.entities_by_sent_id(doc['vertexSet']) == entities_by_sent_id

    def test_relations_by_entities(self):
        assert DocREDUtils.relations_by_entities(doc['labels']) == relations_by_entities