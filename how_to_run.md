# How to Run

## Classification and Evaluation

You can find how to run the classification and evluation script in `run_classification_baseline.sh`.

### Oto
Running with oto example:

CMD: 
```
bash run_classification_baseline.sh
```

Grid: 
```
[{"relation_name": "founded_by", "num_positive_examples": 5, "num_negative_examples": 100},
{"relation_name": "founded_by", "num_positive_examples": 20, "num_negative_examples": 400},
{"relation_name": "founded_by", "num_positive_examples": 50, "num_negative_examples": 1000},
{"relation_name": "founded_by", "num_positive_examples": 100, "num_negative_examples": 2000}]
```

Out Vars: 
```
dev_F1, dev_precision, dev_recall, confidence
```
 (The rest will be saved in RAW)