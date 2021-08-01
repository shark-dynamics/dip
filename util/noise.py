import cv2
import numpy as np
import random
import util.image as um
import matplotlib.pyplot as plt

def random_noise(image, percent):
    '''
    :param image:
    :param percent:
    :return:
    '''
    print('random noise, image shape : {}'.format(image.shape))
    shape = image.shape
    output = np.zeros(shape, np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if random.random() < percent:
                output[i][j] = random.random() * 255.0
            else:
                output[i][j] = image[i][j]
    return output

def random_noise_add(image, percent):
    '''
    :param image:
    :param percent:
    :return:
    '''
    print('random noise, image shape : {}'.format(image.shape))
    shape = image.shape
    output = np.zeros(shape, np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if random.random() < percent:
                output[i][j] = clamp_255(image[i][j] + random.random() * 55.0)
            else:
                output[i][j] = image[i][j]
    return output

def pepper_noise(image, percent=0.05):
    return salt_noise(image ,percent, True)

def salt_noise(image, percent = 0.05, reverse = False):
    shape = image.shape
    output = np.zeros(shape, np.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):
            num = random.random()
            if num < percent:
                if reverse is False:
                    output[i][j] = 255.0
                else:
                    output[i][j] = 0.0
            else:
                output[i][j] = image[i][j]
    return output

def sp_noise(image, percent):
    '''
    :param image:
    :param percent:
    :return: noise image
    '''

    print('sp noise, image shape : {}'.format(image.shape))

    shape = image.shape
    output = np.zeros(shape, np.uint8)

    lower = percent / 2.0
    upper = 1.0 - lower
    for i in range(shape[0]):
        for j in range(shape[1]):
            num = random.random()
            if num < lower:
                output[i][j] = 0.0
                pass
            elif num > upper:
                output[i][j] = 255.0
            else:
                output[i][j] = image[i][j]
    return output

def clamp_255(val):
    if val > 255.0:
        return 255.0
    elif val < 0.0:
        return 0.0
    else:
        return val

def gaussian_noise(image, mean = 0, scale = 20, percent = 0.05):
    '''
    :param image:
    :param percent:
    :return:
    '''
    print('gaussian noise, image shape : {}'.format(image.shape))

    shape = image.shape
    output = np.zeros(shape, np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if random.random() < percent:
                gn = np.random.normal(mean, scale, 1)
                output[i][j] = clamp_255(image[i][j] + gn[0])
            else:
                output[i][j] = image[i][j]
    return output


def noise_test():
    image = np.zeros((300, 300), np.uint8)

    gn = gaussian_noise(image, mean=28, scale=10, percent=0.06)
    x, y = um.get_histogram(gn, except_zero=True)

    plt.figure(figsize=(20,8), dpi=100)

    plt.subplot(241)
    plt.imshow(gn, cmap='gray')
    plt.title('gaussian')
    plt.axis('off')

    gray_hist = plt.subplot(242)
    plt.title('gaussian dist')
    gray_hist.bar(x, y, label="Gray")
    gray_hist.legend()

    sp_img = image + 10
    spn = sp_noise(sp_img, percent=0.06)
    x, y = um.get_histogram(spn, except_value=10)

    plt.subplot(243)
    plt.imshow(spn, cmap='gray')
    plt.title('salt&pepper')
    plt.axis('off')

    gray_hist = plt.subplot(244)
    plt.title('salt&pepper dist')
    gray_hist.bar(x, y, label="Gray")
    gray_hist.legend()

    rn = random_noise(image, percent=0.06)
    x, y = um.get_histogram(rn, except_zero=True)
    plt.subplot(245)
    plt.imshow(rn, cmap='gray')
    plt.title('random')
    plt.axis('off')

    gray_hist = plt.subplot(246)
    plt.title('random dist')
    gray_hist.bar(x, y, label="Gray")
    gray_hist.legend()

    plt.show()