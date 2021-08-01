import cv2
import matplotlib.pyplot as plt
import util.image as um

def get_normalize_histogram(y_values, shape):
    total = shape[0] * shape[1]
    ret = [x*1.0/total for x in y_values]
    return ret

def get_cumulative_normalize_histogram(n_y_values):
    size = len(n_y_values)
    ret = []
    for i in range(size):
        total = 0.0
        for j in range(i+1):
            total += n_y_values[j]
        ret.append(total)
    return ret

def re_create_image(image, sum_n_y_values):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = image[i][j]
            image[i][j] = int(sum_n_y_values[val] * 255 + 0.5)
    return image

def histogram_test():
    img = cv2.imread('images/city.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)
    x_values, y_values = um.get_histogram(img)

    plt.figure(figsize=(8,8), dpi=100)
    gray_hist = plt.subplot(221)
    gray_hist.bar(x_values, y_values, label="Gray")
    gray_hist.legend()

    n_y_values = get_normalize_histogram(y_values, img.shape)
    normal_gray_hist = plt.subplot(222)
    normal_gray_hist.bar(x_values, n_y_values, label="Normalize")
    plt.legend()


    cumulative_n_y_values = get_cumulative_normalize_histogram(n_y_values)
    cumulative_normal_gray_hist = plt.subplot(223)
    cumulative_normal_gray_hist.bar(x_values, cumulative_n_y_values, label="Cumulative Normalize")
    cumulative_normal_gray_hist.legend()

    ret_image = re_create_image(img, cumulative_n_y_values)
    cv2.imshow('histogram image', ret_image)

    x_values, y_values = um.get_histogram(ret_image)

    new_normalize_gray_hist = plt.subplot(224)
    new_normalize_gray_hist.bar(x_values, y_values, label="New Gray")
    new_normalize_gray_hist.legend()

    plt.show()