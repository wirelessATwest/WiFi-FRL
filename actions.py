import numpy as np

def generate_action_set(num_links):
    """
    Return all non-zero binary channel-allocation vectors of length num_links.
    Order matches MATLAB:
    dec2bin(2^Links-1:-1:1)-'0'
    Example for Links=4:
    1111
    1110
    1101
    ...
    0001
    """
    actions = []
    max_state = 2**num_links - 1
    # MATLAB counts down: (2^Links-1) ... 1
    for val in range(max_state, 0, -1):
        bits = [(val >> shift) & 1 for shift in range(num_links-1, -1, -1)]
        actions.append(bits)
    return np.array(actions, dtype=int)
