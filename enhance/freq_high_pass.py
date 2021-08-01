import cv2
import numpy as np
import matplotlib.pyplot as plt
import util.math as um

def high_pass(shape, radius):
    rows, cols = shape
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (int(cols/2), int(rows/2)), radius, 0, -1)
    return mask

def fft_idea_high_pass_test():
    img = cv2.imread("images/test.png", 0)

    mask = high_pass(img.shape, 20)

    mask_i_fft = np.fft.ifft2(mask)
    mask_i_fft = np.fft.ifftshift(mask_i_fft)
    mask_i_fft = np.abs(mask_i_fft)

    mask_info = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mask_info[i][j] = um.max_255(np.abs(int(mask_i_fft[i][j] * 1000000)))

    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(mask, cmap='gray')
    plt.title('mask')
    plt.axis('off')

    plt.subplot(243)
    plt.imshow(mask_info, cmap='gray')
    plt.title('mask space img')
    plt.axis('off')

    fft = np.fft.fft2(img)
    center_fft = np.fft.fftshift(fft)
    center_fft_spec = np.log(np.abs(center_fft))
    plt.subplot(244)
    plt.imshow(center_fft_spec, cmap='gray')
    plt.title('mask space img')
    plt.axis('off')

    mask5 = high_pass(img.shape, 5)
    filter5 = mask5 * center_fft
    filter_img5 = np.abs(np.fft.ifft2(filter5))
    plt.subplot(245)
    plt.imshow(filter_img5, cmap='gray')
    plt.title('radius 5')
    plt.axis('off')

    mask15 = high_pass(img.shape, 15)
    filter15 = mask15 * center_fft
    filter_img15 = np.abs(np.fft.ifft2(filter15))
    plt.subplot(246)
    plt.imshow(filter_img15, cmap='gray')
    plt.title('radius 15')
    plt.axis('off')

    mask30 = high_pass(img.shape, 30)
    filter30 = mask30 * center_fft
    filter_img30 = np.abs(np.fft.ifft2(filter30))
    plt.subplot(247)
    plt.imshow(filter_img30, cmap='gray')
    plt.title('radius 30')
    plt.axis('off')

    mask80 = high_pass(img.shape, 80)
    filter80 = mask80 * center_fft
    filter_img80 = np.abs(np.fft.ifft2(filter80))
    plt.subplot(248)
    plt.imshow(filter_img80, cmap='gray')
    plt.title('radius 80')
    plt.axis('off')

    plt.show()