import argparse
from random import sample
import re

def main(args):
    with open(args.gen_file, 'r') as f:
        gens = f.readlines()

    with open(f'generation_outputs/{args.e1_type}s.txt', 'r') as f:
        e1s = f.readlines()
        e1s = sample(e1s, len(gens))

    with open(f'generation_outputs/{args.e2_type}s.txt', 'r') as f:
        e2s = f.readlines()
        e2s = sample(e2s, len(gens))

    with open(args.gen_file.split('.txt')[0]+'_new_ents.txt', 'w') as f:
        for i, gen in enumerate(gens):
            subbed = re.sub('\[E1\] (.*?) \[\/E1\]', f'[E1] {e1s[i].rstrip()} [/E1]', gen)
            subbed = re.sub('\[E2\] (.*?) \[\/E2\]', f'[E2] {e2s[i].rstrip()} [/E2]', subbed)
            import ipdb; ipdb.set_trace()
            assert subbed != gen
            f.write(subbed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--e1_type", type=str, required=True)
    parser.add_argument("--e2_type", type=str, required=True)
    args = parser.parse_args()
    main(args)