# Bootstrapping Relation Extractors

Implementation of "Bootstrapping Relation Extractors using Syntactic Search by Examples".

## Classification

### Classification and Evaluation

You can find how to run the classification and evluation script in `run_classification.sh`.

##### CMD:
```
bash run_classification.sh
```

Generation
```
{"task": ["tacred"], "training_method": ["generation"], "relation_name": ["org:founded_by"], "num_positive_examples": [100], "ratio_negative_examples": [10], "seed": [1,2,3], "logging_steps": [100]}
```
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
