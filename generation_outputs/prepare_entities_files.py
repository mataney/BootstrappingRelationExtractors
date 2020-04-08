import csv
from tqdm import tqdm
file_paths = ['scripts/search/single_trigger_search_results/raw-ORGANIZATION:PERSON-0']
outfiles = 'generation_outputs/organizations.txt'

entities = set()

for path in file_paths:
    with open (path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        e_index = headers.index('e1')
        for x in tqdm(reader):
            entities.add(x[e_index])


with open(outfiles, 'w') as f:
    for e in entities:
        f.write(f"{e}\n")