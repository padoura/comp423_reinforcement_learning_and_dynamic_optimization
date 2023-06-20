def try_key_initialization(dictionary, key, initial_value):
    if key not in dictionary:
        dictionary[key] = initial_value