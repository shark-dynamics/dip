import cv2
import numpy as np
import util.noise as noise

def inverse(image):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i][j] = 255 - image[i][j]
    return image


def inverse_test():
    img = cv2.imread('images/2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)
    img = inverse(img)
    cv2.imshow('inverse', img)


###################

def segment_en_func(t2, t1, s2, s1, s):
    t = (t2 - t1) * 1.0 / (s2 - s1) * (s - s1) + t1
    return t


def segments_enhance(image, f):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j]
            if 0 <= val < 5:
                image[i][j] = f(0, 0, 5, 0, val)
            if 5 <= val < 120:
                image[i][j] = f(220, 0, 120, 5, val)
            if 120 <= val < 256:
                image[i][j] = f(255, 220, 255, 120, val)

    return image


def segment_enhance_test():
    img = cv2.imread('images/4.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)
    img = segments_enhance(img, segment_en_func)
    cv2.imshow('segment enhance', img)


#####################

def log_en_fun(val):
    return np.log10(1 + val)


def log_enhance(image, f):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j]
            image[i][j] = noise.clamp_255(int(f(val) * 100.0))
    return image


def log_enhance_test():
    img = cv2.imread('images/6.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    img = log_enhance(img, log_en_fun)
    cv2.imshow('log enhance', img)


#################

def r_en_fun(val, r):
    return np.power(val, r)


def r_enhance(image, f, p):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j] * 1.0 / 255.0
            image[i][j] = noise.clamp_255(f(val, p) * 255.0)
    return image


def r_enhance_test():
    origin = cv2.imread('images/6.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', origin)

    img = r_enhance(origin.copy(), r_en_fun, 2.0)
    cv2.imshow('r enhance', img)

    img = r_enhance(origin.copy(), r_en_fun, .2)
    cv2.imshow('r enhance 2', img)


####################

def gray_cut(image, min, max):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j]
            if val < min or val > max:
                image[i][j] = 0
    return image


def gray_cut_test():
    img = cv2.imread('images/moon.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    img = gray_cut(img.copy(), 40, 120)
    cv2.imshow('gray cut', img)


#####################

def threshold_cut(image, th):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j]
            if val < th:
                image[i][j] = 0
            else:
                pass
    return image


def threshold_cut_test():
    img = cv2.imread('images/milk.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)
    img = threshold_cut(img, 120)
    cv2.imshow('threshold', img)


#####################

def bitmap_cut(image):
    shape = image.shape
    result = []
    for i in range(8):
        result.append(np.zeros(shape, np.uint8))

    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j]
            for level in range(8):
                r = int(val / np.power(2, level))
                if r % 2 == 0:
                    result[level][i][j] = 0
                else:
                    result[level][i][j] = 255

    return result


def bitmap_cut_test():
    img = cv2.imread('images/shape.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)
    images = bitmap_cut(img)

    idx = 0
    for image in images:
        cv2.imshow('{}'.format(idx), image)
        idx += 1

#####################