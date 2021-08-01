import cv2
import numpy as np
import matplotlib.pyplot as plt

def drode_test():
    img = cv2.imread('images/dot.png')

    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    # erode
    size = (5,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    eroded_img = cv2.erode(img, kernel, iterations=4)

    plt.subplot(242)
    plt.imshow(eroded_img, cmap='gray')
    plt.title('erode')
    plt.axis('off')

    # dilate
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=4)
    plt.subplot(243)
    plt.imshow(dilated_img, cmap='gray')
    plt.title('dilate')
    plt.axis('off')

    plt.show()