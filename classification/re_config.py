from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, TypeVar
from typing_extensions import Literal, TypedDict

START_E1 = '[E1]'
END_E1 = '[/E1]'
START_E2 = '[E2]'
END_E2 = '[/E2]'

SPECIAL_TOKENS = [START_E1, END_E1, START_E2, END_E2]

RELATIONS_ENTITY_TYPES_FOR_SEARCH = {
    "per:children": "PERSON:PERSON",
    "per:date_of_birth": "PERSON:DATE",
    "org:dissolved": "ORGANIZATION:DATE",
    "org:founded_by": "ORGANIZATION:PERSON",
    "org:country_of_headquarters": "ORGANIZATION:LOCATION",
    "per:country_of_birth": "PERSON:LOCATION",
    "per:religion": "PERSON:MISC",
    "per:spouse": "PERSON:PERSON",
    "per:origin": "PERSON:MISC",
}