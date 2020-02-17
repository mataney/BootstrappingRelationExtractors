import argparse
import pytest

from evaluation import main as evaluation_main

def create_args(pred_file, relation_name):
    args = argparse.Namespace(gold_dir="classification/stubs",
                              gold_file="fake_truth.json",
                              relation_name=relation_name,
                              pred_file=pred_file,
                              confidence_threshold=0,
                              output_file="",
                              ignore_train=False)
    return args

def test_zero():
    args = create_args("classification/stubs/fake_preds0.json", "founded_by")
    scores = evaluation_main(args)
    assert scores['precision'] == 0.0
    assert scores['recall'] == 0.0
    assert scores['F1'] == 0.0

def test_half():
    args = create_args("classification/stubs/fake_preds1.json", "founded_by")
    scores = evaluation_main(args)
    assert scores['precision'] == 1.0
    assert scores['recall'] == 0.5
    assert scores['F1'] == 2/3

def test_full():
    args = create_args("classification/stubs/fake_preds2.json", "founded_by")
    scores = evaluation_main(args)
    assert scores['precision'] == 1.0
    assert scores['recall'] == 1.0
    assert scores['F1'] == 1.0

def test_full_with_diff_evidences():
    args = create_args("classification/stubs/fake_preds3.json", "spouse")
    scores = evaluation_main(args)
    assert scores['precision'] == 1.0
    assert scores['recall'] == 1.0
    assert scores['F1'] == 1.0

def test_two_different_relations():
    args = create_args("classification/stubs/fake_preds4.json", "spouse")
    with pytest.raises(ValueError):
        evaluation_main(args)
    
    args = create_args("classification/stubs/fake_preds4.json", "founded_by")
    with pytest.raises(ValueError):
        evaluation_main(args)

def test_confidence_works():
    args = create_args("classification/stubs/fake_preds5.json", "founded_by")
    scores = evaluation_main(args)
    assert scores['precision'] == 2/3
    assert scores['recall'] == 1.0
    assert scores['F1'] == 0.8

    assert scores['best_precision'] == 1.0
    assert scores['best_recall'] == 1.0
    assert scores['best_F1'] == 1.0
    assert scores['best_confidence'] == 1.0