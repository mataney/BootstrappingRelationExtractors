children_patterns = ["{e1:e=PERSON John} 's [t:w=son|daughter|child|children|daughters|sons daughter] , {e2:e=PERSON Tim}, likes swimming .",
                     "{e1:e=PERSON Mary} did something to her [t:w=son|daughter|child|children|daughters|sons son], {e2:e=PERSON John} in 1992.",
                     "{e1:e=PERSON Mary} was survived by her 4 [t:w=son|daughter|child|children|daughters|sons sons], John, John, {e2:e=PERSON John} and John."]
founded_by_patterns = ["{e1:e=ORGANIZATION Microsoft} [t:w=founder founder] {e2:e=PERSON Mary} likes running.",
                       "{e2:e=PERSON Mary} [t:w=founded founded] {e1:e=ORGANIZATION Microsoft}.",
                       "{e1:e=ORGANIZATION Microsoft} was [t:w=founded founded] [$ by] {e2:e=PERSON Mary}."]
country_of_headquarters_patterns = ["John Doe, a professor at the {e1:e=ORGANIZATION Technion} [in:t=IN in] {e2:e=LOCATION Israel} likes running.",
                                    "{e1:e=ORGANIZATION Technion}, a leading {t:t=/NN/ company} {in:t=IN in} {e2:e=LOCATION Israel}.",
                                    "{e2:e=LOCATION Israel} [pos:t=POS '] largest university is {e1:e=ORGANIZATION BIU}."]
religion_patterns = ["{e1:e=PERSON John} is a [e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Islamist|Islam|Muslim|Shia Jewish]",
                     "[e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Islamist|Islam|Muslim|Shia Jewish] {e1:e=PERSON John} is walking down the street.",
                     "{e1:e=PERSON John} is a [e2:w=Methodist|Episcopal|separatist|Jew|Christian|Sunni|evangelical|atheism|Islamic|secular|fundamentalist|Christianist|Jewish|Anglican|Catholic|orthodox|Scientology|Islamist|Islam|Muslim|Shia Methodist] Person."]
spouse_patterns = ["{e1:e=PERSON John} 's [t:w=wife|husband wife], {e2:e=PERSON Mary} , died in 1991 .",
                   "{e1:e=PERSON John} [t:l=marry married] {e2:e=PERSON Mary}",
                   "{e1:e=PERSON John} is [t:w=married married] to {e2:e=PERSON Mary}"]
origin_patterns = ["{e2:e=MISC Scottish} {e1:e=PERSON Mary} is high.",
                   "{e1:e=PERSON Mary} is a {e2:e=MISC Scottish} professor.",
                   "{e1:e=PERSON Mary}, the {e2:e=LOCATION US} professor."]
date_of_death_patterns = ["{e1:e=PERSON John} was announced [t:w=dead dead] in {e2:e=DATE 1943}.",
                          "{e1:e=PERSON John} [t:w=died died] in {e2:e=DATE 1943}.",
                          "{e1:e=PERSON John}, an NLP scientist, [t:w=died died] {e2:e=DATE 1943}."
        ]
city_of_death_patterns = ["{e1:e=PERSON John} [t:w=died died] in {e2:e=LOCATION London}, {country:e=LOCATION England} in 1997.",
                          "{e1:e=PERSON John} [t:w=died died] in {e2:e=LOCATION London} in 1997.",
                          "{e1:e=PERSON John} [$ -LRB-] [t:w=died died] in {e2:e=LOCATION London} [$ -RRB-] ."]

all_triggers_children_patterns = ["{e1:e=PERSON John} 's [t:w=baby|child|children|daughter|daughters|son|sons|step-daughter|step-son|step-child|step-children|stepchildren|stepdaughter|stepson daughter] , {e2:e=PERSON Tim}, likes swimming .",
                                  "{e1:e=PERSON Mary} did something to her [t:w=baby|child|children|daughter|daughters|son|sons|step-daughter|step-son|step-child|step-children|stepchildren|stepdaughter|stepson son], {e2:e=PERSON John} in 1992.",
                                  "{e1:e=PERSON Mary} was survived by her 4 [t:w=baby|child|children|daughter|daughters|son|sons|step-daughter|step-son|step-child|step-children|stepchildren|stepdaughter|stepson sons], John, John, {e2:e=PERSON John} and John."]
all_triggers_founded_by_patterns = ["{e1:e=ORGANIZATION Microsoft} [t:w=founder|co-founder|cofounder|creator founder] {e2:e=PERSON Mary} likes running.",
                                    "{e2:e=PERSON Mary} [t:w=create|creates|created|creating|creation|co-founded|co-found|debut|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|shapes|shaped|shaping|start|started|starting|starts founded] {e1:e=ORGANIZATION Microsoft}.",
                                    "{e1:e=ORGANIZATION Microsoft} was [t:w=create|creates|created|creating|creation|co-founded|co-found|debut|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|shapes|shaped|shaping|start|started|starting|starts founded] [$ by] {e2:e=PERSON Mary}."]
all_triggers_spouse_patterns = ["{e1:e=PERSON John} 's [t:w=ex-husband|ex-wife|husband|widow|widower|wife|sweetheart|bride wife], {e2:e=PERSON Mary} , died in 1991 .",
                                "{e1:e=PERSON John} [t:w=divorce|divorced|married|marry|wed|divorcing married] {e2:e=PERSON Mary}",
                                "{e1:e=PERSON John} is [t:w=married|marry|wed married] to {e2:e=PERSON Mary}"]
all_triggers_date_of_death_patterns = ["{e1:e=PERSON John} was announced [t:w=dead dead] in {e2:e=DATE 1943}.",
                                       "{e1:e=PERSON John} [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=DATE 1943}.",
                                       "{e1:e=PERSON John}, an NLP scientist, [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] {e2:e=DATE 1943}."]
all_triggers_city_of_death_patterns = ["{e1:e=PERSON John} [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=LOCATION London}, {country:e=LOCATION England} in 1997.",
                                       "{e1:e=PERSON John} [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=LOCATION London} in 1997.",
                                       "{e1:e=PERSON John} [$ -LRB-] [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=LOCATION London} [$ -RRB-] ."]

NEGATIVE_PATTERNS = {
    'PERSON:PERSON': ["(?<e1> [entity=PERSON]+) [entity!=PERSON]+ (?<e2> [entity=PERSON]+) #e e1 e2"],
    'PERSON:DATE': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=DATE]+) #e e1 e2", "(?<e1> [entity=DATE]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
    'ORGANIZATION:DATE': ["(?<e1> [entity=ORGANIZATION]+) []+ (?<e2> [entity=DATE]+) #e e1 e2", "(?<e1> [entity=DATE]+) []+ (?<e2> [entity=ORGANIZATION]+) #e e1 e2"],
    'ORGANIZATION:PERSON': ["(?<e1> [entity=ORGANIZATION]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2", "(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=ORGANIZATION]+) #e e1 e2"],
    'ORGANIZATION:LOCATION': ["(?<e1> [entity=ORGANIZATION]+) []+ (?<e2> [entity=LOCATION]+) #e e1 e2", "(?<e1> [entity=LOCATION]+) []+ (?<e2> [entity=ORGANIZATION]+) #e e1 e2"],
    'PERSON:LOCATION': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=LOCATION]+) #e e1 e2", "(?<e1> [entity=LOCATION]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
    'PERSON:MISC': ["(?<e1> [entity=PERSON]+) []+ (?<e2> [entity=MISC]+) #e e1 e2", "(?<e1> [entity=MISC]+) []+ (?<e2> [entity=PERSON]+) #e e1 e2"],
}


docred_founded_by_patterns = ["{e1:e=ORGANIZATION|MISC Microsoft} [t:w=founder founder] {e2:e=PERSON Mary} likes running.",
                       "{e2:e=PERSON Mary} [t:w=founded founded] {e1:e=ORGANIZATION|MISC Microsoft}.",
                       "{e1:e=ORGANIZATION|MISC Microsoft} was [t:w=founded founded] [$ by] {e2:e=PERSON Mary}."]
docred_origin_patterns = ["{e2:e=MISC Scottish} company, {e1:e=ORGANIZATION Microsoft} is successful.",
                          "{e1:e=ORGANIZATION|MISC Microsoft} is a {e2:e=MISC Scottish} Company.",
                          "{e1:e=ORGANIZATION|MISC Microsoft} is a {t:t=/NN/ song} [$ by] {e2:e=MISC Scottish} musican."]
docred_date_of_death_patterns = ["{e1:e=PERSON John} [$ -LRB-]  [$:e=DATE date] [$ -] {e2:e=DATE 1997} [$ -RRB-] .",
                                 "{e1:e=PERSON John} [t:w=died died] in {e2:e=DATE 1943}.",
                                 "{e1:e=PERSON John}, an NLP scientist, [t:w=died died] {e2:e=DATE 1943}."]
docred_city_of_death_patterns = ["{e1:e=PERSON John} [t:w=died died] in {e2:e=LOCATION London}, {country:e=LOCATION England} in 1997.",
                                 "{e1:e=PERSON John} [t:w=died died] in {e2:e=LOCATION London} in 1997.",
                                 "{e1:e=PERSON John} [$ -LRB-] [$:e=DATE 1997], [$:e=LOCATION London] [$ -] [$:e=DATE 1997] {e2:e=LOCATION London} [$ -RRB-] ."]
docred_country_of_headquarters_patterns =  ["{e1:e=ORGANIZATION Technion}, a leading {t:t=/NN/ company} {in:t=IN in} {e2:e=LOCATION Israel}.",
                                            "{e1:e=ORGANIZATION Microsoft} is [t:l=base|headquarter based] in {e2:e=LOCATION England} .",
                                            "{e1:e=ORGANIZATION Technion}, a leading {t:t=/NN/ company} based {in:t=IN in} {e2:e=LOCATION Israel}."]

all_triggers_docred_founded_by_patterns = ["{e1:e=ORGANIZATION|MISC Microsoft} [t:w=founder|co-founder|cofounder|creator founder] {e2:e=PERSON Mary} likes running.",
                                            "{e2:e=PERSON Mary} [t:w=create|creates|created|creating|creation|co-founded|co-found|debut|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|shapes|shaped|shaping|start|started|starting|starts founded] {e1:e=ORGANIZATION|MISC Microsoft}.",
                                            "{e1:e=ORGANIZATION|MISC Microsoft} was [t:w=create|creates|created|creating|creation|co-founded|co-found|debut|emerge|emerges|emerged|emerging|establish|established|establishing|establishes|establishment|forge|forges|forged|forging|forms|formed|forming|founds|found|founded|founding|launched|launches|launching|opened|opens|opening|shapes|shaped|shaping|start|started|starting|starts founded] [$ by] {e2:e=PERSON Mary}."]
all_triggers_docred_date_of_death_patterns = ["{e1:e=PERSON John} [$ -LRB-]  [$:e=DATE date] [$ -] {e2:e=DATE 1997} [$ -RRB-] .",
                                 "{e1:e=PERSON John} [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=DATE 1943}.",
                                 "{e1:e=PERSON John}, an NLP scientist, [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] {e2:e=DATE 1943}."]
all_triggers_docred_city_of_death_patterns = ["{e1:e=PERSON John} [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=LOCATION London}, {country:e=LOCATION England} in 1997.",
                                 "{e1:e=PERSON John} [t:w=died|executed|killed|dies|perished|succumbed|passed|murdered|suicided died] in {e2:e=LOCATION London} in 1997.",
                                 "{e1:e=PERSON John} [$ -LRB-] [$:e=DATE 1997], [$:e=LOCATION London] [$ -] [$:e=DATE 1997] {e2:e=LOCATION London} [$ -RRB-] ."]

SINGLE_TRIGGER_PATTERNS = {
    'tacred': {
        "per:children": children_patterns,
        "org:founded_by": founded_by_patterns,
        "org:country_of_headquarters": country_of_headquarters_patterns,
        "per:religion": religion_patterns,
        "per:spouse": spouse_patterns,
        "per:origin": origin_patterns,
        "per:date_of_death": date_of_death_patterns,
        "per:city_of_death": city_of_death_patterns,
    },
    'docred': {
        "per:children": children_patterns,
        "org:founded_by": docred_founded_by_patterns,
        "org:country_of_headquarters": docred_country_of_headquarters_patterns,
        "per:religion": religion_patterns,
        "per:spouse": spouse_patterns,
        "per:origin": docred_origin_patterns,
        "per:date_of_death": docred_date_of_death_patterns,
        "per:city_of_death": docred_city_of_death_patterns,
    },
}

ALL_TRIGGERS_PATTERNS = {
    'tacred': {
        "per:children": all_triggers_children_patterns,
        "org:founded_by": all_triggers_founded_by_patterns,
        "per:spouse": all_triggers_spouse_patterns,
        "per:date_of_death": all_triggers_date_of_death_patterns,
        "per:city_of_death": all_triggers_city_of_death_patterns,
    },
    'docred': {
        "per:children": all_triggers_children_patterns,
        "org:founded_by": all_triggers_docred_founded_by_patterns,
        "per:spouse": all_triggers_spouse_patterns,
        "per:date_of_death": all_triggers_docred_date_of_death_patterns,
        "per:city_of_death": all_triggers_docred_city_of_death_patterns,
    },
}