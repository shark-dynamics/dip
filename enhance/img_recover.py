import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2


def gen_motion_blur_direction(shape, angel, length=15):
    psf = np.zeros(shape)
    x = math.cos( np.radians(angel) ) * length
    y = math.sin( np.radians(angel) ) * length
    center = (int(shape[1]/2), int(shape[0]/2))
    next_point = (int(center[0] + x), int(center[1] + y))
    cv2.line(psf, center, next_point, 1)
    return psf

def gen_motion_blur_image(image, psf):
    input_fft = fft.fft2(image)
    psf_fft = fft.fft2(psf)
    blur_img = fft.ifft2(input_fft * psf_fft)
    blur_img = np.abs(fft.fftshift(blur_img))
    return blur_img, np.abs(psf_fft)


def inverse_filter(image, psf):
    input_fft = fft.fft2(image)
    psf_fft = fft.fft2(psf)
    ### psf 中有0的话，会出错，加一点点偏移值保证不为0
    inverse_img = fft.ifft2(input_fft / (psf_fft + 0.0001))
    inverse_img = np.abs(fft.fftshift(inverse_img))
    return inverse_img


def wiener_filter(image, psf, K=0.01):
    img_fft = fft.fft2(image)
    psf_fft = fft.fft2(psf)
    loss_func = psf_fft / (np.abs((psf_fft)) ** 2 + K)
    filter_img = fft.ifft2(img_fft * loss_func)
    filter_img = np.abs(fft.fftshift(filter_img))
    return filter_img


def motion_blur_recover_test():

    image = cv2.imread('images/moon_gray.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(8, 6), dpi=120)
    plt.gray()
    psf = gen_motion_blur_direction(image.shape, 45, 16)
    motion_blur_img, psf_fft = gen_motion_blur_image(image, psf)

    plt.subplot(341)
    plt.imshow(image, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    plt.subplot(342)
    plt.title("psf")
    plt.imshow(psf, cmap='gray')
    plt.axis('off')

    plt.subplot(343)
    plt.title("psf fft")
    plt.imshow(psf_fft, cmap='gray')
    plt.axis('off')

    plt.subplot(345)
    plt.title("blur")
    plt.imshow(motion_blur_img, cmap='gray')
    plt.axis('off')

    inverse_filter_img = inverse_filter(motion_blur_img, psf)
    plt.subplot(346)
    plt.title("inverse img")
    plt.imshow(inverse_filter_img, cmap='gray')
    plt.axis('off')

    #其实K直接取0也无所谓，退化为inverse filter 也可以，因为psf是自己构造的,无需矫正
    wiener_filter_img = wiener_filter(motion_blur_img, psf, 0.001)
    plt.subplot(347)
    plt.title("wiener img")
    plt.imshow(wiener_filter_img, cmap='gray')
    plt.axis('off')

    blurred_noisy = motion_blur_img + 80 * np.random.standard_normal(motion_blur_img.shape)  # 添加噪声,standard_normal产生随机的函数
    plt.subplot2grid((3, 4), (2, 0), colspan=1, rowspan=1)
    plt.title("motion noise")
    plt.imshow(blurred_noisy, cmap='gray')
    plt.axis('off')

    result = inverse_filter(blurred_noisy, psf)  # 对添加噪声的图像进行逆滤波
    plt.subplot2grid((3, 4), (2, 1), colspan=1, rowspan=1)
    plt.title("inverse img")
    plt.imshow(result)
    plt.axis('off')

    result = wiener_filter(blurred_noisy, psf, 1.01)  # 对添加噪声的图像进行维纳滤波
    plt.subplot2grid((3, 4), (2, 2), colspan=1, rowspan=1)
    plt.title("wiener img")
    plt.imshow(result)
    plt.axis('off')

    plt.show()
