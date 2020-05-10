import argparse
import json

def pos_logging_steps_ratio(training_method, pos):
    if training_method == 'generation':
        return 200
    pos_logging_steps_ratio = 5
    ret = pos*pos_logging_steps_ratio
    if ret > 500:
        ret = 200
    elif ret > 100:
        ret = 100
    return ret

def get_ratio_negative(training_method, relation):
    # ratio_negative_examples = {"per:children": 47, "org:founded_by": 22, "org:country_of_headquarters": 4, "per:religion": 2, "per:spouse": 29, "per:origin": 5, "per:date_of_death": 8, "per:city_of_death": 5, "child": 85, "date_of_death": 12, "founded_by": 247, "religion": 470, "spouse": 78, "country_of_origin": 57, "headquarters_location": 71, "place_of_death": 85}

    # if training_method == "search":
        # return 10
    return 10

def get_num_positives(training_method, relation):
    num_positive_examples = {"annotated": [5, 10, 20, 50, 100], "search": [100, 500, 1000], "distant": [100, 500, 1000], "generation": [100], "search_from_generation": [100]}
    #max isn't necessarily correct
    max_train_examples = {
        # tacred
        "per:children": 211, "org:founded_by": 124, "org:country_of_headquarters": 468, "per:religion": 53, "per:spouse": 258, "per:origin": 325, "per:date_of_death": 134, "per:city_of_death": 81,
        # docred
        "child": 296, "founded_by": 70, "headquarters_location": 208, "religion": 121, "spouse": 238, "country_of_origin": 429, "date_of_death": 650, "place_of_death": 158,
        }

    default = num_positive_examples[training_method]
    if training_method == "annotated":
        if max_train_examples[relation] > default[-1]:
            return default + [max_train_examples[relation]]
    return default

seeds = [1, 2, 3]

ALL_RELATION_NAMES = {"tacred": ["per:children", "org:founded_by", "org:country_of_headquarters",
                                 "per:religion", "per:spouse", "per:origin", "per:date_of_death", "per:city_of_death"],
                      "docred": ["child", "date_of_death", "founded_by", "religion", "spouse",
                                 "country_of_origin", "headquarters_location", "place_of_death"]}

def main(args):
    all_params = []
    for task in args.tasks:
        if args.relation_names == ['all']:
            args.relation_names = ALL_RELATION_NAMES[task]
        for relation in args.relation_names:
            assert relation in ALL_RELATION_NAMES[task]
            for training_method in args.training_methods:
                for num_positive in get_num_positives(training_method, relation):
                    ratio_negative = get_ratio_negative(training_method, relation)
                    for seed in seeds:
                        all_params.append({"task": task, "training_method": training_method, "relation_name": relation, "num_positive_examples": num_positive, "ratio_negative_examples": ratio_negative, "logging_steps": pos_logging_steps_ratio(training_method, num_positive), "seed": seed})
    
    print(json.dumps(all_params).replace('}, {', '},\n {'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--tasks',
                        nargs='+',
                        type=str,
                        required=True,
                        choices=['tacred', 'docred'])
    parser.add_argument('--training_methods',
                        nargs='+',
                        type=str,
                        required=True,
                        choices=['annotated', 'distant', 'search', 'generation', 'search_from_generation'])
    parser.add_argument('--relation_names',
                        nargs='+',
                        type=str,
                        required=True,
                        choices=["per:children", "org:founded_by", "org:country_of_headquarters", "per:religion", "per:spouse", "per:origin", "per:date_of_death", "per:city_of_death", 
                        "child", "date_of_death", "founded_by", "religion", "spouse", "country_of_origin", "headquarters_location", "place_of_death",
                        "all"])

    args = parser.parse_args()
    main(args)