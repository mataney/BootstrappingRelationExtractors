from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, TypeVar
from typing_extensions import Literal, TypedDict

START_E1 = '[E1]'
END_E1 = '[/E1]'
START_E2 = '[E2]'
END_E2 = '[/E2]'

SPECIAL_TOKENS = [START_E1, END_E1, START_E2, END_E2]