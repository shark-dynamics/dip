import numpy as np

import util.math


def plus(img_a, img_b, reverse=False):
    shape = img_a.shape
    output = np.zeros(shape, np.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):
            src = int(img_a[i][j])
            dst = int(img_b[i][j])
            if reverse is True:
                output[i][j] = np.abs(src - dst)
            else:
                output[i][j] = util.math.max_255(src+dst)
    return output

def minus(img_a, img_b):
    return plus(img_a, img_b, True)


def get_histogram(image, except_zero=False, except_value=None):
    values = {}
    for i in range(256):
        values[i] = 0

    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            v = image[i][j]
            if except_zero is True and v == 0:
                continue
            if except_value is not None and v == except_value:
                continue
            values[v] += 1

    x_values = []
    y_values = []
    for k, v in values.items():
        x_values.append(k)
        y_values.append(v)

    return x_values, y_values