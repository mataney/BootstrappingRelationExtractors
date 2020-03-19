RELATION_MAPPING = {'org:founded_by': {'id': 'org:founded_by', 'subj_type': ['ORGANIZATION'], 'obj_type': ['PERSON']}, \
    'per:employee_of': {'id': 'per:employee_of', 'subj_type': ['PERSON'], 'obj_type': ['ORGANIZATION']}, \
    'org:alternate_names': {'id': 'org:alternate_names', 'subj_type': ['ORGANIZATION'], 'obj_type': ['ORGANIZATION']}, \
    'per:cities_of_residence': {'id': 'per:cities_of_residence', 'subj_type': ['PERSON'], 'obj_type': ['CITY', 'LOCATION']}, \
    'per:children': {'id': 'per:children', 'subj_type': ['PERSON'], 'obj_type': ['PERSON']}, \
    'per:title': {'id': 'per:title', 'subj_type': ['PERSON'], 'obj_type': ['TITLE']}, \
    'per:siblings': {'id': 'per:siblings', 'subj_type': ['PERSON'], 'obj_type': ['PERSON']}, \
    'per:religion': {'id': 'per:religion', 'subj_type': ['PERSON'], 'obj_type': ['RELIGION']}, \
    'per:age': {'id': 'per:age', 'subj_type': ['PERSON'], 'obj_type': ['NUMBER', 'DURATION']}, \
    'org:website': {'id': 'org:website', 'subj_type': ['ORGANIZATION'], 'obj_type': ['URL']}, \
    'per:stateorprovinces_of_residence': {'id': 'per:stateorprovinces_of_residence', 'subj_type': ['PERSON'], 'obj_type': ['STATE_OR_PROVINCE']}, \
    'org:member_of': {'id': 'org:member_of', 'subj_type': ['ORGANIZATION'], 'obj_type': ['ORGANIZATION', 'COUNTRY']}, \
    'org:top_members/employees': {'id': 'org:top_members/employees', 'subj_type': ['ORGANIZATION'], 'obj_type': ['PERSON']}, \
    'per:countries_of_residence': {'id': 'per:countries_of_residence', 'subj_type': ['PERSON'], 'obj_type': ['COUNTRY', 'NATIONALITY']}, \
    'org:city_of_headquarters': {'id': 'org:city_of_headquarters', 'subj_type': ['ORGANIZATION'], 'obj_type': ['CITY']}, \
    'org:members': {'id': 'org:members', 'subj_type': ['ORGANIZATION'], 'obj_type': ['ORGANIZATION', 'COUNTRY']}, \
    'org:country_of_headquarters': {'id': 'org:country_of_headquarters', 'subj_type': ['ORGANIZATION'], 'obj_type': ['COUNTRY']}, \
    'per:spouse': {'id': 'per:spouse', 'subj_type': ['PERSON'], 'obj_type': ['PERSON']}, \
    'org:stateorprovince_of_headquarters': {'id': 'org:stateorprovince_of_headquarters', 'subj_type': ['ORGANIZATION'], 'obj_type': ['STATE_OR_PROVINCE']}, \
    'org:number_of_employees/members': {'id': 'org:number_of_employees/members', 'subj_type': ['ORGANIZATION'], 'obj_type': ['NUMBER']}, \
    'org:parents': {'id': 'org:parents', 'subj_type': ['ORGANIZATION'], 'obj_type': ['ORGANIZATION']}, \
    'org:subsidiaries': {'id': 'org:subsidiaries', 'subj_type': ['ORGANIZATION'], 'obj_type': ['ORGANIZATION']}, \
    'per:origin': {'id': 'per:origin', 'subj_type': ['PERSON'], 'obj_type': ['COUNTRY', 'NATIONALITY']}, \
    'org:political/religious_affiliation': {'id': 'org:political/religious_affiliation', 'subj_type': ['ORGANIZATION'], 'obj_type': ['RELIGION', 'IDEOLOGY']}, \
    'per:other_family': {'id': 'per:other_family', 'subj_type': ['PERSON'], 'obj_type': ['PERSON']}, \
    'per:stateorprovince_of_birth': {'id': 'per:stateorprovince_of_birth', 'subj_type': ['PERSON'], 'obj_type': ['STATE_OR_PROVINCE']}, \
    'org:dissolved': {'id': 'org:dissolved', 'subj_type': ['ORGANIZATION'], 'obj_type': ['DATE']}, \
    'per:date_of_death': {'id': 'per:date_of_death', 'subj_type': ['PERSON'], 'obj_type': ['DATE']}, \
    'org:shareholders': {'id': 'org:shareholders', 'subj_type': ['ORGANIZATION'], 'obj_type': ['PERSON', 'ORGANIZATION']}, \
    'per:alternate_names': {'id': 'per:alternate_names', 'subj_type': ['PERSON'], 'obj_type': ['PERSON']}, \
    'per:parents': {'id': 'per:parents', 'subj_type': ['PERSON'], 'obj_type': ['PERSON']}, \
    'per:schools_attended': {'id': 'per:schools_attended', 'subj_type': ['PERSON'], 'obj_type': ['ORGANIZATION']}, \
    'per:cause_of_death': {'id': 'per:cause_of_death', 'subj_type': ['PERSON'], 'obj_type': ['CAUSE_OF_DEATH']}, \
    'per:city_of_death': {'id': 'per:city_of_death', 'subj_type': ['PERSON'], 'obj_type': ['CITY']}, \
    'per:stateorprovince_of_death': {'id': 'per:stateorprovince_of_death', 'subj_type': ['PERSON'], 'obj_type': ['STATE_OR_PROVINCE']}, \
    'org:founded': {'id': 'org:founded', 'subj_type': ['ORGANIZATION'], 'obj_type': ['DATE']}, \
    'per:country_of_birth': {'id': 'per:country_of_birth', 'subj_type': ['PERSON'], 'obj_type': ['COUNTRY']}, \
    'per:date_of_birth': {'id': 'per:date_of_birth', 'subj_type': ['PERSON'], 'obj_type': ['DATE']}, \
    'per:city_of_birth': {'id': 'per:city_of_birth', 'subj_type': ['PERSON'], 'obj_type': ['CITY']}, \
    'per:charges': {'id': 'per:charges', 'subj_type': ['PERSON'], 'obj_type': ['CRIMINAL_CHARGE']}, \
    'per:country_of_death': {'id': 'per:country_of_death', 'subj_type': ['PERSON'], 'obj_type': ['COUNTRY', 'NATIONALITY', 'LOCATION']},
    'no_relation': {'id': 'no_relation'}}