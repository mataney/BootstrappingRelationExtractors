# Relation Generation Using GPT2

In thie repository we use [huggingface's pytorch-transformers](https://github.com/huggingface/pytorch-transformers/) to generate relation examples given a user input. Therefore, pytorch-transformers should be either installed using `pip` or you should add the clone path to your `$PYTHONPATH`.

## Finetune

You should finetune on your dataset using a `run_lm_finetuning.py` or an easy to use bash script similar to the one used for TACRED `tacred_generation.sh`. This file is also an example of the arguments you should pass `run_lm_finetuning.py`.

## Generation

After finetuning, pass the model alongside different hyperparameters to `run_generation.py`. This should also recieve a sentence in the prompet like the following: `William married Kate Middleton. <|GO|>`. Again, you can find an example of the arguments in the corresponding bash script `tacred_generation`.

## Filtering

You should filter out examples using premade trigger list. This might be expended to other cases as well. Use `trigger_filtering.py` script.

## Cluster and Rank syntactic patterns

This happens when connecting to spike, example coming soon. (For now, like in `diversity evaluation method.ipynb`)