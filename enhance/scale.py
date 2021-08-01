import cv2
import matplotlib.pyplot as plt

def scale_test():
    img = cv2.imread('images/2.jpg', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(241)
    plt.imshow(img)
    plt.title('origin')
    plt.axis('off')

    shape = img.shape[:2]
    small = cv2.resize(img.copy(), (int(shape[0]/3), int(shape[1]/3)))
    plt.subplot(242)
    plt.imshow(small)
    plt.title('small')
    plt.axis('off')

    big = cv2.resize(small.copy(), shape, interpolation=cv2.INTER_NEAREST)
    plt.subplot(243)
    plt.imshow(big)
    plt.title('nearest')
    plt.axis('off')

    big = cv2.resize(small.copy(), shape, interpolation=cv2.INTER_CUBIC)
    plt.subplot(244)
    plt.imshow(big)
    plt.title('cubic')
    plt.axis('off')

    big = cv2.resize(small.copy(), shape, interpolation=cv2.INTER_LINEAR)
    plt.subplot(245)
    plt.imshow(big)
    plt.title('linear')
    plt.axis('off')

    big = cv2.resize(small.copy(), shape, interpolation=cv2.INTER_LANCZOS4)
    plt.subplot(246)
    plt.imshow(big)
    plt.title('lanczos')
    plt.axis('off')

    plt.show()