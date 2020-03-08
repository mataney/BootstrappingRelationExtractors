import json

from classification.tacred import TACREDProcessor, TACREDExample

data_dir = 'data/tacred'

with open('classification/stubs/tacred/fake_truth.json', "r", encoding="utf-8") as f:
    fakes = list(json.load(f))

class TestTACREDProcessor:
    def test_get_train_examples(self):
        processor = TACREDProcessor('org:founded_by', 5, 2)
        examples = processor.get_examples_by_set_type('train', data_dir)
        assert len(examples) == 5 + 5 * 2
        assert len([e for e in examples if e.label == "org:founded_by"]) == 5
        assert len([e for e in examples if e.label != "org:founded_by"]) == 10

    def test_get_dev_examples(self):
        processor = TACREDProcessor('org:founded_by', 5, 2)
        examples = processor.get_examples_by_set_type('full_test_eval', data_dir)
        assert len([e for e in examples if e.label == "org:founded_by"]) == 68
        assert len([e for e in examples if e.label != "org:founded_by"]) == 1190
        assert len(examples) == 68 + 1190

class TestTACREDExample:
    def test_init(self):
        fake = fakes[0]
        example = TACREDExample(0, fake, "org:founded_by")
        assert example.id == 0
        assert example.label == "org:founded_by"
        assert example.text.startswith("[E2] Tom Thabane [/E2] resigned in October last year to form the [E1] All Basotho Convention [/E1]")