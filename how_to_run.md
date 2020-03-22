# How to Run

## Classification and Evaluation

You can find how to run the classification and evluation script in `run_classification.sh`.

### Oto
Running with oto example:

#### CMD:
```
bash run_classification.sh
```

#### Out Vars:
```
test_F1, test_precision, test_recall, confidence, time
```
 (The rest will be saved in RAW)

#### Define Arguments
Explicitly:
```
[{"task": "docred", "relation_name": "founded_by", "num_positive_examples": 5, "ratio_negative_examples": 10},
{"task": "docred", "relation_name": "founded_by", "num_positive_examples": 20, "ratio_negative_examples": 10}]
```

Running classification for all relations as a grid for docred:
```
{"task": ["docred"], "relation_name": ["child", "date_of_death", "founded_by", "religion", "spouse", "country_of_origin", "headquarters_location", "place_of_death"], "num_positive_examples": [5,10,20,100], "ratio_negative_examples": [10], "seed": [1,2,3]}
```

Or for Distant Supervision:
```
{"task": ["docred"], "training_method": ["distant"], "relation_name": ["child", "date_of_death", "founded_by", "religion", "spouse", "country_of_origin", "headquarters_location", "place_of_death"], "num_positive_examples": [100, 500, 1000], "ratio_negative_examples": [10], "seed": [1,2,3]}
```

Running classification for all relations as a grid for TACRED:
```
{"task": ["tacred"], "relation_name": ["per:children", "per:date_of_death", "org:founded_by", "per:religion", "per:spouse", "per:origin", "org:country_of_headquarters", "per:city_of_death"], "num_positive_examples": [5,10,20,100], "ratio_negative_examples": [10], "seed": [1,2,3]}
```

 ### Explore Errors

 You can run `expore_error_types.py` to explore the outputs of your model, just copy the RAW from Oto. Then, you can run the analyzing in the following way:

 ```
 python expore_error_types.py --raw raw_file --confidence_threshold threshold --report tp fp fn
 ```

 Where `threshold` should be the best dev threshold, and `report` should by one or more from `[tp, fp, fn]` that stand for "true positive", "false poritive" and "false negative'.