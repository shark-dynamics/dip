import cv2
import util.math as um
import util.noise as noise
import util.image as iu
import numpy as np

def filter_image(image, kernel, kernel_coefficient = 0.0):
    size = len(kernel)
    print('kernel size : {}'.format(size))
    print('shape : {}'.format(image.shape))
    offset = -int((size-1)/2)

    if kernel_coefficient != 0:
        for i in range(size):
            for j in range(size):
                kernel[i][j] *= kernel_coefficient

    shape = image.shape
    output = np.zeros(shape, np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            total = 0.0
            for x in range(size):
                xOffset = offset + x
                for y in range(size):
                    yOffset = offset + y
                    # k = kernel[x][y]
                    k = kernel[y][x]
                    image_i_idx = um.max_threshold(um.min_0(i + yOffset), shape[0]-1)
                    image_j_idx = um.max_threshold(um.min_0(j + xOffset), shape[1]-1)
                    val = image[image_i_idx][image_j_idx]
                    total += k * val
            # if less than 0, change to 0...
            output[i][j] = um.max_255(um.min_0(int(total)))

    return output

def make_avg_kernel(size):
    avg_value = 1.0/(size*size)
    avg_kernel = []
    for i in range(size):
        kernel_line = []
        for j in range(size):
            kernel_line.append(avg_value)
        avg_kernel.append(kernel_line)
    return avg_kernel

def filter_average_test():
    img = cv2.imread('images/2.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    avg_kernel = make_avg_kernel(3)
    avg_image = filter_image(img.copy(), avg_kernel)
    cv2.imshow('K3', avg_image)

    # avg_kernel = make_avg_kernel(5)
    # avg_image = filter_image(img.copy(), avg_kernel)
    # cv2.imshow('K5', avg_image)
    #
    # avg_kernel = make_avg_kernel(7)
    # avg_image = filter_image(img.copy(), avg_kernel)
    # cv2.imshow('k7', avg_image)
    #
    # avg_kernel = make_avg_kernel(9)
    # avg_image = filter_image(img.copy(), avg_kernel)
    # cv2.imshow('k9', avg_image)

def filter_weight_avg_test():
    img = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    coefficient = 1.0/16.0
    wa_kernel = [
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0]]

    fm = filter_image(img.copy(), wa_kernel, coefficient)
    cv2.imshow('fm', fm)

############################################

def filter_middle_func(values):
    values.sort()
    return values[int(len(values)/2)]

def filter_max_func(values):
    values.sort(reverse=True)
    return values[0]

def filter_min_func(values):
    values.sort()
    return values[0]

def filter_mid_value_func(values):
    values.sort()
    length = len(values)
    start = int(values[0])
    end = int(values[length-1])
    v = int((start+end)/2)
    # print('values : {} v : {}'.format(values, v))
    return v

def filter_image_non_linear(image, kernel, func):
    size = len(kernel)
    offset = -int((size-1)/2)

    shape = image.shape

    # pixel_values = []
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         val = image[i][j]
    #         pixel_values.append(val)
    # print('pixel vale : {}'.format(pixel_values))

    output = np.zeros(shape, np.uint8)

    for i in range(shape[0]):
        for j in range(shape[1]):
            values = []
            for x in range(size):
                xOffset = offset + x
                for y in range(size):
                    yOffset = offset + y

                    image_i_idx = um.max_threshold(um.min_0(i + yOffset), shape[0]-1)
                    image_j_idx = um.max_threshold(um.min_0(j + xOffset), shape[1]-1)
                    val = image[image_i_idx][image_j_idx]
                    values.append(val)
            output[i][j] = int(func(values))

    return output

def filter_image_mid_value_test():
    img = cv2.imread('images/5.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    noise_img = noise.sp_noise(img.copy(), 0.05)
    cv2.imshow('SP noise', noise_img)

    kernel = make_avg_kernel(3)
    fm = filter_image_non_linear(noise_img.copy(), kernel, filter_middle_func)
    cv2.imshow('filter SP', fm)

    noise_img = noise.random_noise(img.copy(), 0.05)
    cv2.imshow('Random Noise', noise_img)

    fm = filter_image_non_linear(noise_img.copy(), kernel, filter_middle_func)
    cv2.imshow('filter Random', fm)

def filter_image_max_min_value_test():
    img = cv2.imread('images/moon.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    salt_noise_img = noise.salt_noise(img.copy())
    cv2.imshow('salt noise', salt_noise_img)

    kernel = make_avg_kernel(3)
    f_min_img = filter_image_non_linear(salt_noise_img.copy(), kernel, filter_min_func)
    cv2.imshow('filter salt', f_min_img)

    pepper_noise_img = noise.pepper_noise(img.copy())
    cv2.imshow('pepper noise', pepper_noise_img)

    f_max_img = filter_image_non_linear(pepper_noise_img, kernel, filter_max_func)
    cv2.imshow('filter pepper', f_max_img)

def filter_middle_value_test():
    img = cv2.imread('images/4.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    r_img = noise.random_noise_add(img, 0.65)
    cv2.imshow('r img', r_img)

    f_mid_val_img = filter_image_non_linear(r_img.copy(), make_avg_kernel(3), filter_mid_value_func)
    cv2.imshow('filter mid value', f_mid_val_img)


def filter_sobel_image(image, kernel1, kernel2):
    size = len(kernel1)
    offset = -int((size-1)/2)

    shape = image.shape
    output = np.zeros(shape, np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            total_x = 0.0
            total_y = 0.0
            for x in range(size):
                xOffset = offset + x
                for y in range(size):
                    yOffset = offset + y
                    k1 = kernel1[y][x]
                    k2 = kernel2[y][x]

                    image_i_idx = um.max_threshold(um.min_0(i + yOffset), shape[0]-1)
                    image_j_idx = um.max_threshold(um.min_0(j + xOffset), shape[1]-1)
                    val = image[image_i_idx][image_j_idx]

                    total_x += k1 * val
                    total_y += k2 * val
                    # print('k1 : {} k2 : {}, val : {}'.format(k1, k2, val))
            # print('total x : {}, total y : {}'.format(total_x, total_y))
            # output[i][j] = um.max_255(int( np.abs(total_x)) + int( np.abs(total_y) ))
            output[i][j] = um.max_255(int(np.sqrt(total_x*total_x + total_y*total_y)))

    return output

def filter_sobel_test():
    img = cv2.imread('images/5.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    kernel1 = [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0]]

    kernel2 = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]]

    fm = filter_sobel_image(img.copy(), kernel1, kernel2)
    cv2.imshow('sobel', fm)

    r = cv2.Sobel(img, -1, 1, 1, ksize=3)
    cv2.imshow('S', r)

def filter_laplace_test():
    img = cv2.imread('images/moon2.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    # kernel = [
    #     [0.0, 1.0, 0.0],
    #     [1.0, -4.0, 1.0],
    #     [0.0, 1.0, 0.0]]

    kernel = [
        [0.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 0.0]]

    fm = filter_image(img.copy(), kernel)
    cv2.imshow('laplace+', fm)

    enhance = iu.plus(img, fm)
    cv2.imshow('enhance', enhance)

    l = cv2.Laplacian(img, -1)
    cv2.imshow('OpenCV L', l)















