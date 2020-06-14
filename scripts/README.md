# Scripts

## search/download_search_examples.py

This scripts downloads from spike all the sentences that share the same syntactic pattern as patterns you defined in the top of the file. This is done by passing `--download` flag.

After downloading, you need to merge all of these file into 2 single files, one for positive examples and another for negatives. This can be done by running `--merge_patterns`.

So if I want to download and merge all files for all patterns of all relations just run:

```
python -m scripts.search.download_search_examples --merge_patterns --triggers single --dataset tacred
```

`--triggers single` means I'm using only a single relation for each pattern. can also pass `all`.
`--dataset tacred` saves it in the `data/tacred` directory.

## search/patterns_from_generation.py

Is a script that given an annotated generation for a specific relation file finds for each generation it's syntactic rule and downloads a sample of the examples in this pattern.

```
python -m scripts.search.patterns_from_generation --generation_file generation_outputs/with_triggers_for_search_using_generation/per:children.txt --relation per:children --dataset tacred --download_explanations --download_examples --merge_patterns
```

It first start with finding the explanations (syntactic rules) for each genration (using the `--download_explanations` flag) then it continues to download the sample of corresponding examples (using the `--download_flag` flag) and then it merges the downloads similarly to `download_search_examples.py`

You can also evaluate the examples you downloaded with the `--evaluate` flag: 

```
python -m scripts.search.patterns_from_generation --generation_file generation_outputs/finished_files/with_triggers_for_search_using_generation/per:children.txt --relation per:children --dataset tacred --evaluate
```