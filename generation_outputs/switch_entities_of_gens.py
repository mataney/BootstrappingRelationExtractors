import argparse
from itertools import product
from random import sample
import re
from tqdm import tqdm

PERSONAL_PRONOUNS_TO_KEEP = ['he', 'she']
POSSESIVE_PRONOUNS_TO_KEEP = ['his', 'her']
ENTITY_TYPES = {
         'children': ['person', 'person', 'person', None],
         'city_of_death': ['person', 'city', 'city', None],
         'date_of_death': ['person', 'date', 'person', None],
         'founded_by': ['organization', 'person', 'person', 'organization'],
         'origin-country': ['person', 'country', 'person', 'country'],
         'origin-nationality': ['person', 'nationality', 'person', 'nationality'],
         'religion': ['person', 'religion', 'person', 'religion'],
         'spouse': ['person', 'person', 'person', None],
        }

def main(args):
    entity_types = ENTITY_TYPES[args.relation]
    with open(args.gen_file, 'r') as f:
        gens = f.readlines()
    gens = [g for g in gens if g!= '\n']

    e1s = get_similar_entities(entity_types[0])
    e2s = get_similar_entities(entity_types[1])
    e3s = get_similar_entities(entity_types[2])
    e4s = get_similar_entities(entity_types[3])

    with open(args.gen_file.split('.txt')[0]+'_new_ents.txt', 'w') as f:
        for i, gen in tqdm(enumerate(gens)):
            assert gen.count('[') == gen.count(']'), gen
            # E1 - PERSON/ORGANIZATION
            subbed = switch_entity_but_not_pronouns(1, gen, e1s)
            # E2
            if entity_types[1] == 'date':
                subbed = switch_dates(2, subbed, e2s)
            elif entity_types[1] == 'religion':
                subbed = switch_religions(2, subbed, e2s)
            else:
                subbed = switch_entity_but_not_pronouns(2, subbed, e2s)
            # E3 - PERSON/ORGANIZATION
            if entity_types[2]:
                for e in re.findall('\[E3\] (.*?) \[\/E3\]', subbed):
                    subbed = re.sub(f'\[E3\] {e} \[\/E3\]', sample(e3s, 1)[0], subbed)
            # E4
            if entity_types[3]:
                for e in re.findall('\[E4\] (.*?) \[\/E4\]', subbed):
                    if entity_types[3] == 'religion':
                        subbed = switch_religions(4, subbed, e4s, keep_markers=False)
                    else:
                        subbed = re.sub(f'\[E4\] {e} \[\/E4\]', sample(e4s, 1)[0], subbed)
            if subbed == gen:
                print(f"Warning, generation didn't change: {gen}")
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

def switch_religions(ent_num, subbed, ents, keep_markers=True):
    E = f"E{ent_num}"
    if keep_markers:
        subbed = re.sub(f"\[{E}\] Religion \[\/{E}\]", f"[{E}] {sample(ents['religion'], 1)[0]} [/{E}]", subbed)
        subbed = re.sub(f"\[{E}\] Religious Affiliation \[\/{E}\]", f"[{E}] {sample(ents['religious_affiliation'], 1)[0]} [/{E}]", subbed)
        subbed = re.sub(f"\[{E}\] Religious Relation \[\/{E}\]", f"[{E}] {sample(ents['religious_relation'], 1)[0]} [/{E}]", subbed)
        subbed = re.sub(f"\[{E}\] Religious Affiliation plural \[\/{E}\]", f"[{E}] {sample(ents['religious_relation'], 1)[0]} [/{E}]", subbed)
    else:
        subbed = re.sub(f"\[{E}\] Religion \[\/{E}\]", sample(ents['religion'], 1)[0], subbed)
        subbed = re.sub(f"\[{E}\] Religious Affiliation \[\/{E}\]", sample(ents['religious_affiliation'], 1)[0], subbed)
        subbed = re.sub(f"\[{E}\] Religious Relation \[\/{E}\]", sample(ents['religious_relation'], 1)[0], subbed)
        subbed = re.sub(f"\[{E}\] Religious Affiliation plural \[\/{E}\]", sample(ents['religious_relation'], 1)[0], subbed)
    return subbed

def switch_dates(ent_num, subbed, ents):
    # changing just november 7.
    #TODO, pretty sure this is depracated. Why only November 7?
    E = f"E{ent_num}"
    subbed = re.sub(f'\[{E}\] November 7 \[\/{E}\]', f'[{E}] {sample(ents, 1)[0]} [/{E}]', subbed)
    subbed = re.sub(f'\[{E}\] Nov 7 \[\/{E}\]', f'[{E}] {sample(ents, 1)[0]} [/{E}]', subbed)
    subbed = re.sub(f'\[{E}\] Nov\. 7 \[\/{E}\]', f'[{E}] {sample(ents, 1)[0]} [/{E}]', subbed)

def get_similar_entities(entity_type):
    if entity_type is None:
        return None
    elif entity_type == 'date':
        return dates()
    elif entity_type == 'city':
        return cities()
    elif entity_type == 'nationality':
        return nationalities()
    elif entity_type == 'religion':
        return religions()
    else:
        return read_ents_from_file(entity_type)

def read_ents_from_file(entity_type):
    with open(f'generation_outputs/types/{entity_type}.txt', 'r') as f:
        ents = f.readlines()
        ents = [e.rstrip() for e in ents]
    return ents

def religions():
    religions = {
        'religion': ["Atheism", "Scientology", "Islam", "Christianity"],
        'religious_relation': ["Evangelical", "Islamic", "Christian", "Jewish", "Catholic"],
        'religious_affiliation': ["Methodist", "Separatist", "Jew", "Christian", "Sunni", "Secular", "Fundamentalist", "Christianist", "Anglican", "Orthodox", "Conservative", "Islamist", "Muslim"],
    }
    religions['religious_affiliation_plural'] = [f"{x}s" for x in religions['religious_affiliation']]

    return religions

def dates():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return [' '.join([x[0], str(x[1])]) for x in product(months, range(29))]

def cities():
    return ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "San Francisco", "Charlotte", "Indianapolis", "Seattle", "Denver", "Washington", "Boston", "El Paso", "Detroit", "Nashville", "Portland", "Memphis", "Oklahoma City", "Las Vegas", "Louisville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Mesa", "Sacramento", "Atlanta", "Kansas City", "Colorado Springs", "Miami", "Raleigh", "Omaha", "Long Beach", "Virginia Beach", "Oakland", "Minneapolis", "Tulsa", "Arlington", "Tampa", "New Orleans"]

def nationalities():
    return ["British", "English", "Scottish", "Gaelic", "Irish", "Welsh", "Danish", "Finnish", "Norwegian", "Swedish", "Swiss", "German", "French", "Italian", "Estonian", "Latvian", "Lithuanian", "Austrian", "Belgian", "Flemish", "Dutch", "American", "Canadian", "Mexican", "Spanish", "Ukrainian", "Russian", "Belarusian", "Polish", "Czech", "Slovak", "Slovakian", "Hungarian", "Romanian", "Bulgarian", "Greek", "Brazilian", "Portuguese", "Australian", "New Zealander", "Maori", "Georgian", "Israeli", "Hebrew", "Egyptian", "Arabic", "Turkish", "Chinese", "Mandarin", "Korean", "Japanese", "Indian", "Hindi", "South African", "Afrikaans"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file", type=str, required=True)
    parser.add_argument("--relation", type=str, required=True)
    args = parser.parse_args()
    main(args)