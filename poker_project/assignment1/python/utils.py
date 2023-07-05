def try_key_initialization(dictionary, key, initial_value):
    ''' Initialize key in dictionary only if it does not already exist
    '''
    if key not in dictionary:
        dictionary[key] = initial_value

def get_moving_average(arr, window_size):
    i = 1
    window_sum = sum(arr[:window_size])
    window_average = window_sum / window_size
    moving_averages = [window_average]
    while i < len(arr) - window_size + 1:
        
        # Update window sum by subtracting first element and adding new element
        window_sum += arr[i + window_size - 1] - arr[i-1]
    
        # Calculate the average of current window
        window_average = window_sum / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        
        # Shift window to right by one position
        i += 1
    
    return moving_averages

def get_random_max_key(dictionary, np_random):
    ''' Randomly selects one of the keys that possess the max value
    '''
    max_value = max(dictionary.values())
    max_keys = [key for key, value in dictionary.items() if value == max_value]
    return np_random.choice(max_keys)