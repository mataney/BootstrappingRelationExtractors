import json
import os
import sys

import torch

def main(file_dir, write_to):
    out = {}
    for file in os.listdir(file_dir):
        if file.startswith("cached"):
            examples = torch.load(os.path.join(file_dir, file))
            pos = len([e for e in examples if e.label == 0])
            neg = len([e for e in examples if e.label == 1])
            out[file[7:file.index('_roberta')]] = {"num_pos": pos, "num_neg": neg}

    json.dump(out, open(write_to, 'w'))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])