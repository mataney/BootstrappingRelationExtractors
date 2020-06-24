# Relation Generation Using GPT2

## Classification

### Classification and Evaluation

You can find how to run the classification and evluation script in `run_classification.sh`.

#### Oto
Running with oto example:

##### CMD:
```
bash run_classification.sh
bash tacred_finetuning.sh
```

##### Out Vars:
```
test_F1, test_precision, test_recall, confidence, time
```
 (The rest will be saved in RAW)

##### Define Arguments
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

Search Single
Need to change if Single or All.
If All, need to add "per:religion".
```
{"task": ["tacred"], "training_method": ["search"], "relation_name": ["per:children", "per:date_of_death", "org:founded_by", "per:spouse", "per:origin", "org:country_of_headquarters", "per:city_of_death"], "num_positive_examples": [100, 500, 1000], "ratio_negative_examples": [10], "seed": [1,2,3]}
```

Generation
```
{"task": ["tacred"], "training_method": ["generation"], "relation_name": ["org:founded_by"], "num_positive_examples": [100], "ratio_negative_examples": [10], "seed": [1,2,3], "logging_steps": [100]}

 #### Explore Errors

 You can run `expore_error_types.py` to explore the outputs of your model, just copy the RAW from Oto. Then, you can run the analyzing in the following way:

 ```
 python expore_error_types.py --raw raw_file --confidence_threshold threshold --report tp fp fn
 ```

 Where `threshold` should be the best dev threshold, and `report` should by one or more from `[tp, fp, fn]` that stand for "true positive", "false poritive" and "false negative'.

## Generation

Here I'm mostly using modified scripts of huggingface's transformers.

### Preprocessing
In order to create the trainable examples run
```
python preprocess/create_tacred_datafiles.py --file_path ../datasets/tacred/data/json/train.json --save_to_file data/tacred/for_generation/train --src_and_tgt_one_file_with_go
```

### Finetune

You should finetune on your dataset using a `run_lm_finetuning.py` or an easy to use bash script similar to the one used for TACRED `tacred_generation.sh`. This file is also an example of the arguments you should pass `run_lm_finetuning.py`.

### Generation

After finetuning, pass the model alongside different hyperparameters to `run_generation.py`. This should also recieve a sentence in the prompet like the following: `William married Kate Middleton. <|GO|>`. Again, you can find an example of the arguments in the corresponding bash script `tacred_generation`.

### Filtering

You should filter out examples using premade trigger list. This might be expended to other cases as well. Use `trigger_filtering.py` script.

### Cluster and Rank syntactic patterns

This happens when connecting to spike, example coming soon. (For now, like in `diversity evaluation method.ipynb`)