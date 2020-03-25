from scripts.search.download_search_examples import seperate_entities, SearchSortedListMonotonicIncreasingVal

def populate_data(values):
    return {'sentence_id': '1',
            'e1_first_index': values[0],
            'e1_last_index': values[1],
            'e2_first_index': values[2],
            'e2_last_index': values[3]}

def test_seperate_entities_all_e1_before_e2():
    data = populate_data([1, 3, 5, 6])
    assert seperate_entities(data)

def test_seperate_entities_all_e2_before_e1():
    data = populate_data([10, 11, 5, 6])
    assert seperate_entities(data)

def test_seperate_entities_some_e1_before_e2_some_not():
    data = populate_data([1, 3, 3, 6])
    assert not seperate_entities(data)

    data = populate_data([1, 4, 3, 6])
    assert not seperate_entities(data)

def test_seperate_entities_some_e2_before_e1_some_not():
    data = populate_data([3, 6, 1, 3])
    assert not seperate_entities(data)

    data = populate_data([3, 6, 1, 4])
    assert not seperate_entities(data)

def test_seperate_entities_e1_equal_to_e2():
    data = populate_data([1, 3, 1, 3])
    assert not seperate_entities(data)

def test_seperate_entities_e1_before_and_after_e2():
    data = populate_data([1, 6, 5, 6])
    assert not seperate_entities(data)

    data = populate_data([1, 300, 5, 6])
    assert not seperate_entities(data)

def test_seperate_entities_e2_before_and_after_e1():
    data = populate_data([5, 6, 1, 6])
    assert not seperate_entities(data)

    data = populate_data([5, 6, 1, 300])
    assert not seperate_entities(data)
