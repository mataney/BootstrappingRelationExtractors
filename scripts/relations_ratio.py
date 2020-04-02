from math import ceil
from classification.tacred import TACREDProcessor
from classification.docred import DocREDProcessor

dataset = 'TACRED'

if dataset == 'TACRED':
    relation_names = ["per:children", "org:founded_by", "org:country_of_headquarters", "per:religion", "per:spouse", "per:origin", "per:date_of_death", "per:city_of_death"]
    for relation_name in relation_names:
        num_positive = 100000000
        negative_ratio = 100000000
        type_independent_neg_sample = False
        processor = TACREDProcessor(relation_name, num_positive, negative_ratio, type_independent_neg_sample)
        examples = processor.get_examples_by_set_type('full_dev_eval', 'data/tacred')
        positives = len([e for e in examples if e.label == relation_name])
        negatives = len([e for e in examples if e.label == 'NOTA'])
        assert positives + negatives == len(examples)
        print(f"{relation_name}: {ceil(negatives / positives)}")
elif dataset == 'DocRED':
    relation_names = ["child", "date_of_death", "founded_by", "religion", "spouse", "country_of_origin", "headquarters_location", "place_of_death"]
    for relation_name in relation_names:
        num_positive = 100000000
        negative_ratio = 100000000
        type_independent_neg_sample = False
        processor = DocREDProcessor(relation_name, num_positive, negative_ratio, type_independent_neg_sample)
        examples = processor.get_examples_by_set_type('full_dev_eval', 'data/DocRED')
        positives = len([e for e in examples if e.label == relation_name])
        negatives = len([e for e in examples if e.label == 'NOTA'])
        assert positives + negatives == len(examples)
        print(f"{relation_name}: {ceil(negatives / positives)}")
elif:
    print("Wrong dataset name")