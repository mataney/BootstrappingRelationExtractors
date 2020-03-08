# How to Run

## Classification and Evaluation

You can find how to run the classification and evluation script in `run_classification.sh`.

### Oto
Running with oto example:

#### CMD:
```
bash run_classification.sh
```

#### Define Arguments
Explicitly:
```
[{"relation_name": "founded_by", "num_positive_examples": 5, "ratio_negative_examples": 10},
{"relation_name": "founded_by", "num_positive_examples": 20, "ratio_negative_examples": 10}]
```

Running classification for all relations as a grid:
```
{"relation_name": ["child", "country_of_origin", "date_of_birth", "dissolved,_abolished_or_demolished", "founded_by", "headquarters_location", "place_of_birth", "religion", "spouse"], "num_positive_examples": [5,10,20,100], "ratio_negative_examples": [10], "seed": [1,2,3]}
```

Or for Distant Supervision:
```
{"training_method": ["distant"], "relation_name": ["child", "country_of_origin", "date_of_birth", "dissolved,_abolished_or_demolished", "founded_by", "headquarters_location", "place_of_birth", "religion", "spouse"], "num_positive_examples": [100, 500, 1000], "ratio_negative_examples": [10], "logging_steps": [1000]}
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

 Where `threshold` should be the best dev threshold, and `report` should by one or more from `[tp, fp, fn]` that stand for "true positive", "false poritive" and "false negative'.