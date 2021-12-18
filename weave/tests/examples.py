"""Example data for tests."""
from pandas import DataFrame


# Input data
data = DataFrame({
    'name': ['five', 'five', 'six', 'seven', 'nine'],
    'location_id': [5, 5, 6, 7, 9],
    'year_id': [1980, 1990, 2000, 2010, 2020],
    'age_mid': [1.0, 2.0, 3.0, 4.0, 5.0],
    'count': [100, 200, 300, 400, 500],
    'rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'level_1': [1, 1, 1, 1, 2],
    'level_2': [3, 3, 3, 4, 8],
    'level_3': [5, 5, 6, 7, 9],
    'holdout': [False, True, True, False, False]
})

# Hierarchy levels
levels = ['level_1', 'level_2', 'level_3']

# Lists of wrong types and/or values to test exceptions
test_dict = {
    'int': [1.0, 'dummy', True, None, [], (), {}],
    'float': [1, 'dummy', True, None, [], (), {}],
    'numeric': ['dummy', True, None, [], (), {}],
    'str': [1, 1.0, True, None, [], (), {}],
    'list': [1, 1.0, 'dummy', True, None, (), {}],
    'other': [1, 1.0, 'dummy', True, None, [], (), {}]
}
test_dict['list_of_str'] = test_dict['other'] + \
                           [[not_str] for not_str in test_dict['str']]
