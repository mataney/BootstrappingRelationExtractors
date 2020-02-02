START_E1 = '[E1]'
END_E1 = '[/E1]'
START_E2 = '[E2]'
END_E2 = '[/E2]'

SPECIAL_TOKENS = [START_E1, END_E1, START_E2, END_E2]

CLASS_MAPPING = {'headquarters location': {'id': 'P159', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'country': {'id': 'P17', 'e1_type': 'MISC', 'e2_type': 'LOC'}, 'located in the administrative territorial entity': {'id': 'P131', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'contains administrative territorial entity': {'id': 'P150', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'country of citizenship': {'id': 'P27', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'date of birth': {'id': 'P569', 'e1_type': 'PER', 'e2_type': 'TIME'}, 'place of birth': {'id': 'P19', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'inception': {'id': 'P571', 'e1_type': 'ORG', 'e2_type': 'TIME'}, 'dissolved, abolished or demolished': {'id': 'P576', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'located in or next to body of water': {'id': 'P206', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'has part': {'id': 'P527', 'e1_type': 'ORG', 'e2_type': 'PER'}, 'member of': {'id': 'P463', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'performer': {'id': 'P175', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'publication date': {'id': 'P577', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'place of death': {'id': 'P20', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'date of death': {'id': 'P570', 'e1_type': 'PER', 'e2_type': 'TIME'}, 'part of': {'id': 'P361', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'capital of': {'id': 'P1376', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'capital': {'id': 'P36', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'spouse': {'id': 'P26', 'e1_type': 'PER', 'e2_type': 'PER'}, 'mother': {'id': 'P25', 'e1_type': 'PER', 'e2_type': 'PER'}, 'father': {'id': 'P22', 'e1_type': 'PER', 'e2_type': 'PER'}, 'child': {'id': 'P40', 'e1_type': 'PER', 'e2_type': 'PER'}, 'country of origin': {'id': 'P495', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'developer': {'id': 'P178', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'platform': {'id': 'P400', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'member of political party': {'id': 'P102', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'point in time': {'id': 'P585', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'location of formation': {'id': 'P740', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'record label': {'id': 'P264', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'conflict': {'id': 'P607', 'e1_type': 'ORG', 'e2_type': 'MISC'}, 'educated at': {'id': 'P69', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'production company': {'id': 'P272', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'employer': {'id': 'P108', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'work location': {'id': 'P937', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'military branch': {'id': 'P241', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'position held': {'id': 'P39', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'languages spoken, written or signed': {'id': 'P1412', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'composer': {'id': 'P86', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'participant of': {'id': 'P1344', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'location': {'id': 'P276', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'lyrics by': {'id': 'P676', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'member of sports team': {'id': 'P54', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'notable work': {'id': 'P800', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'author': {'id': 'P50', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'narrative location': {'id': 'P840', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'present in work': {'id': 'P1441', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'characters': {'id': 'P674', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'original network': {'id': 'P449', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'genre': {'id': 'P136', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'legislative body': {'id': 'P194', 'e1_type': 'LOC', 'e2_type': 'ORG'}, 'applies to jurisdiction': {'id': 'P1001', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'owned by': {'id': 'P127', 'e1_type': 'LOC', 'e2_type': 'ORG'}, 'located on terrain feature': {'id': 'P706', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'producer': {'id': 'P162', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'continent': {'id': 'P30', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'participant': {'id': 'P710', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'sibling': {'id': 'P3373', 'e1_type': 'PER', 'e2_type': 'PER'}, 'head of state': {'id': 'P35', 'e1_type': 'LOC', 'e2_type': 'PER'}, 'territory claimed by': {'id': 'P1336', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'award received': {'id': 'P166', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'residence': {'id': 'P551', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'head of government': {'id': 'P6', 'e1_type': 'LOC', 'e2_type': 'PER'}, 'director': {'id': 'P57', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'screenwriter': {'id': 'P58', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'league': {'id': 'P118', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'mouth of the watercourse': {'id': 'P403', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'subclass of': {'id': 'P279', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'end time': {'id': 'P582', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'start time': {'id': 'P580', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'creator': {'id': 'P170', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'operator': {'id': 'P137', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'publisher': {'id': 'P123', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'followed by': {'id': 'P156', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'follows': {'id': 'P155', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'cast member': {'id': 'P161', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'part of the series': {'id': 'P179', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'chairperson': {'id': 'P488', 'e1_type': 'ORG', 'e2_type': 'PER'}, 'instance of': {'id': 'P31', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'manufacturer': {'id': 'P176', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'subsidiary': {'id': 'P355', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'founded by': {'id': 'P112', 'e1_type': 'ORG', 'e2_type': 'PER'}, 'official language': {'id': 'P37', 'e1_type': 'LOC', 'e2_type': 'MISC'}, 'ethnic group': {'id': 'P172', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'unemployment rate': {'id': 'P1198', 'e1_type': 'LOC', 'e2_type': 'NUM'}, 'influenced by': {'id': 'P737', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'original language of performance work': {'id': 'P364', 'e1_type': 'MISC', 'e2_type': 'LOC'}, 'religion': {'id': 'P140', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'basin country': {'id': 'P205', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'parent organization': {'id': 'P749', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'product or material produced': {'id': 'P1056', 'e1_type': 'ORG', 'e2_type': 'MISC'}, 'replaces': {'id': 'P1365', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'parent taxon': {'id': 'P171', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'replaced by': {'id': 'P1366', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'separated from': {'id': 'P807', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'twinned administrative body': {'id': 'P190', 'e1_type': 'LOC', 'e2_type': 'LOC'}}