import cv2
import numpy as np
import util.noise as noise

def add_mean(images):
    shape = images[0].shape
    output = np.zeros(shape, np.float)
    for img in images:
        for i in range(shape[0]):
            for j in range(shape[1]):
                output[i][j] += (img[i][j]*1.0 / 255.0)

    image = np.zeros(shape, np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = output[i][j] * 1.0 / len(images) * 255.0
            image[i][j] = int(val)
    return image

def add_meat_test():
    img = cv2.imread('images/moon.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', img)

    noise_image_tips = None
    images = []
    for i in range(20):
        noise_img = noise.sp_noise(img, 0.05)
        # noise_img = noise.gaussian_noise(img)
        if noise_image_tips is None:
            noise_image_tips = noise_img
        images.append(noise_img)
    whole_out = add_mean(images)
    cv2.imshow('20 result', whole_out)
    cv2.imshow('noise', noise_image_tips)

    half_image_result = add_mean(images[10:])
    cv2.imshow('10 result', half_image_result)