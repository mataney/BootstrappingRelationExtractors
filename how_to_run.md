# How to Run

## Classification and Evaluation

You can find how to run the classification and evluation script in `run_classification_baseline.sh`.

### Oto
Running with oto example:

#### CMD:
```
bash run_classification_baseline.sh
```

#### Define Arguments
Explicitly:
```
[{"relation_name": "founded_by", "num_positive_examples": 5, "num_negative_examples": 100},
{"relation_name": "founded_by", "num_positive_examples": 20, "num_negative_examples": 400},
{"relation_name": "founded_by", "num_positive_examples": 50, "num_negative_examples": 1000},
{"relation_name": "founded_by", "num_positive_examples": 100, "num_negative_examples": 2000}]
```

or as a grid:
```
{"relation_name": ["religion"], "num_positive_examples": [5,10,20,100], "ratio_negative_examples": [5,10,20], "seed": [1,2,3]}
```

#### Out Vars:
```
dev_F1, dev_precision, dev_recall, confidence
```
 (The rest will be saved in RAW)

 ### Explore Errors

 You can run `expore_error_types.py` to explore the outputs of your model, just copy the RAW from Oto. Then, you can run the analyzing in the following way:

 ```
 python expore_error_types.py --raw raw_file --confidence_threshold threshold --report tp fp fn
 ```

 Where `threshold` should be the best train_eval threshold, and `report` should by one or more from `[tp, fp, fn]` that stand for "true positive", "false poritive" and "false negative'.