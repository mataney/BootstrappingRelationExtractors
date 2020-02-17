from collections import defaultdict
import json
import os
from random import shuffle

data_dir = "data/DocRED/"
data_file =  "train_annotated.json"
with open(os.path.join(data_dir, data_file), 'r') as f:
    data = json.load(f)

shuffle(data)

bar = int(len(data)*0.8)
train_split, eval_split = data[:bar], data[bar:]

assert len(train_split) + len(eval_split) == len(data)

with open(os.path.join(data_dir, 'train_split_from_annotated.json'), 'w') as outfile:
    json.dump(train_split, outfile)

with open(os.path.join(data_dir, 'eval_split_from_annotated.json'), 'w') as outfile:
    json.dump(eval_split, outfile)