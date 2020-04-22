import argparse
from itertools import product
from random import sample
import re

PERSONAL_PRONOUNS_TO_KEEP = ['he', 'she']
POSSESIVE_PRONOUNS_TO_KEEP = ['his', 'her']
ENTITY_TYPES = {
         'children': ['person', 'person', 'person', None],
         'city_of_death': ['person', 'city', 'city', None],
         'date_of_death': ['person', 'date', 'person', None],
         'founded_by': ['organization', 'person', 'person', 'organization'],
        }

def main(args):
    entity_types = ENTITY_TYPES[args.relation]
    with open(args.gen_file, 'r') as f:
        gens = f.readlines()
    gens = [g for g in gens if g!= '\n']

    with open(f'generation_outputs/types/{entity_types[0]}.txt', 'r') as f:
        e1s = f.readlines()
        e1s = [e.rstrip() for e in e1s]

    if entity_types[1] == 'date':
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        e2s = [' '.join([x[0], str(x[1])]) for x in product(months, range(29))]
    elif entity_types[1] == 'city':
        e2s = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco", "Charlotte", "Indianapolis", "Seattle", "Denver", "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland", "Memphis", "Oklahoma City", "Las Vegas", "Louisville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Mesa", "Sacramento", "Atlanta", "Kansas City", "Colorado Springs", "Miami", "Raleigh", "Omaha", "Long Beach", "Virginia Beach", "Oakland", "Minneapolis", "Tulsa", "Arlington", "Tampa", "New Orleans"]
    else:
        with open(f'generation_outputs/types/{entity_types[1]}.txt', 'r') as f:
            e2s = f.readlines()
            e2s = [e.rstrip() for e in e2s]

    if entity_types[2]:
        with open(f'generation_outputs/types/{entity_types[2]}.txt', 'r') as f:
            e3s = f.readlines()
            e3s = [e.rstrip() for e in e3s]

    if entity_types[3]:
        with open(f'generation_outputs/types/{entity_types[3]}.txt', 'r') as f:
            e4s = f.readlines()
            e4s = [e.rstrip() for e in e4s]

    with open(args.gen_file.split('.txt')[0]+'_new_ents.txt', 'w') as f:
        for i, gen in enumerate(gens):
            subbed = switch_entity_but_not_pronouns(1, gen, e1s)
            if entity_types[1] == 'date':
                #changing just november 7
                subbed = re.sub('\[E2\] November 7 \[\/E2\]', f'[E2] {sample(e2s, 1)[0]} [/E2]', subbed)
                subbed = re.sub('\[E2\] Nov 7 \[\/E2\]', f'[E2] {sample(e2s, 1)[0]} [/E2]', subbed)
                subbed = re.sub('\[E2\] Nov\. 7 \[\/E2\]', f'[E2] {sample(e2s, 1)[0]} [/E2]', subbed)
            else:
                subbed = switch_entity_but_not_pronouns(2, subbed, e2s)
            if entity_types[2]:
                for e in re.findall('\[E3\] (.*?) \[\/E3\]', subbed):
                    subbed = re.sub(f'\[E3\] {e} \[\/E3\]', sample(e3s, 1)[0], subbed)
            if entity_types[3]:
                for e in re.findall('\[E4\] (.*?) \[\/E4\]', subbed):
                    subbed = re.sub(f'\[E4\] {e} \[\/E4\]', sample(e4s, 1)[0], subbed)
            assert subbed != gen
            f.write(subbed)

def switch_entity_but_not_pronouns(ent_num, gen, ents):
    E = f"E{ent_num}"
    found_pronouns = re.findall(f"\[{E}\] ({'|'.join(PERSONAL_PRONOUNS_TO_KEEP+POSSESIVE_PRONOUNS_TO_KEEP)}) \[\/{E}\]", gen, flags=re.IGNORECASE)
    if found_pronouns:
        for p in re.findall(f"\[{E}\] ({'|'.join(PERSONAL_PRONOUNS_TO_KEEP)}) \[\/{E}\]", gen, flags=re.IGNORECASE):
            gen = re.sub(f'\[{E}\] ({p}) \[\/{E}\]', f'[{E}] {sample(PERSONAL_PRONOUNS_TO_KEEP, 1)[0]} [/{E}]', gen, flags=re.IGNORECASE)
        for p in re.findall(f"\[{E}\] ({'|'.join(POSSESIVE_PRONOUNS_TO_KEEP)}) \[\/{E}\]", gen, flags=re.IGNORECASE):
            gen = re.sub(f'\[{E}\] ({p}) \[\/{E}\]', f'[{E}] {sample(POSSESIVE_PRONOUNS_TO_KEEP, 1)[0]} [/{E}]', gen, flags=re.IGNORECASE)
    else:
        gen = re.sub(f'\[{E}\] (.*?) \[\/{E}\]', f'[{E}] {sample(ents, 1)[0]} [/{E}]', gen)

    return gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    args = parser.parse_args()
    main(args)