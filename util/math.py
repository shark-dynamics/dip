def min_0(val):
    if val < 0:
        return 0
    return val

def max_threshold(val, th):
    if val > th:
        return th
    return val

def max_255(val):
    return max_threshold(val, 255)