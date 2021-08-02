import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_freq_images(image):
    fft_img = np.fft.fft2(image)
    shift_fft_img = np.fft.fftshift(fft_img.copy())
    return fft_img, shift_fft_img

def change_to_int(image):
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = image[i][j]
            output[i][j] = int(val)
    return output

def fft_test():

    img = cv2.imread("images/3.jpg", 0)
    fft = np.fft.fft2(img)
    center_fft = np.fft.fftshift(fft)

    inverse_fft_img = np.fft.ifft2(fft)
    inverse_fft_img = np.abs(inverse_fft_img)

    origin_img_spectrum = np.log(np.abs(fft))

    box_size = 25
    rows, cols = img.shape
    center_row, center_col = int(rows/2), int(cols/2)
    high_pass_fft = center_fft.copy()
    high_pass_fft[center_row-box_size:center_row+box_size, center_col-box_size:center_col+box_size] = 0
    high_pass_fft_spectrum = np.log(np.abs(high_pass_fft))
    high_pass_img = np.fft.ifft2(high_pass_fft)
    high_pass_img = np.abs(high_pass_img)

    low_pass_fft = center_fft.copy()
    mask = np.zeros(img.shape, np.uint8)
    mask[center_row-box_size:center_row+box_size, center_col-box_size:center_col+box_size] = 1
    low_pass_fft *= mask
    low_pass_fft_spectrum = np.log(np.abs(low_pass_fft))
    low_pass_img = np.fft.ifft2(low_pass_fft)
    low_pass_img = np.abs(low_pass_img)

    center_img_spectrum = np.log(np.abs(center_fft))
    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(origin_img_spectrum, cmap='gray')
    plt.title('origin spec')
    plt.axis('off')

    plt.subplot(243)
    plt.imshow(center_img_spectrum, cmap='gray')
    plt.title("center spec")
    plt.axis('off')

    plt.subplot(244)
    plt.imshow(inverse_fft_img, cmap='gray')
    plt.title("inverse fft img")
    plt.axis('off')

    plt.subplot(245)
    plt.imshow(high_pass_fft_spectrum, cmap='gray')
    plt.title("high pass filter")
    plt.axis('off')

    plt.subplot(246)
    plt.imshow(high_pass_img, cmap='gray')
    plt.title("high pass img")
    plt.axis('off')

    plt.subplot(247)
    plt.imshow(low_pass_fft_spectrum, cmap='gray')
    plt.title("low pass filter")
    plt.axis('off')

    plt.subplot(248)
    plt.imshow(low_pass_img, cmap='gray')
    plt.title("low pass img")
    plt.axis('off')

    plt.show()

def low_pass(shape, radius):
    rows, cols = shape
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (int(cols/2), int(rows/2)), radius, 1, -1)
    return mask

def fft_idea_low_pass_test():
    img = cv2.imread("images/test.png", 0)

    mask = low_pass(img.shape, 6)

    mask_i_fft = np.fft.ifft2(mask)
    mask_i_fft = np.fft.ifftshift(mask_i_fft)
    mask_i_fft = np.abs(mask_i_fft)

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
    plt.imshow(mask_i_fft, cmap='gray')
    plt.title('mask space img')
    plt.axis('off')

    fft = np.fft.fft2(img)
    center_fft = np.fft.fftshift(fft)
    center_fft_spec = np.log(np.abs(center_fft))
    plt.subplot(244)
    plt.imshow(center_fft_spec, cmap='gray')
    plt.title('img freq')
    plt.axis('off')

    mask5 = low_pass(img.shape, 5)
    filter5 = mask5 * center_fft
    filter_img5 = np.abs(np.fft.ifft2(filter5))
    plt.subplot(245)
    plt.imshow(filter_img5, cmap='gray')
    plt.title('radius 5')
    plt.axis('off')

    mask15 = low_pass(img.shape, 15)
    filter15 = mask15 * center_fft
    filter_img15 = np.abs(np.fft.ifft2(filter15))
    plt.subplot(246)
    plt.imshow(filter_img15, cmap='gray')
    plt.title('radius 15')
    plt.axis('off')

    mask30 = low_pass(img.shape, 30)
    filter30 = mask30 * center_fft
    filter_img30 = np.abs(np.fft.ifft2(filter30))
    plt.subplot(247)
    plt.imshow(filter_img30, cmap='gray')
    plt.title('radius 30')
    plt.axis('off')

    mask80 = low_pass(img.shape, 80)
    filter80 = mask80 * center_fft
    filter_img80 = np.abs(np.fft.ifft2(filter80))
    plt.subplot(248)
    plt.imshow(filter_img80, cmap='gray')
    plt.title('radius 80')
    plt.axis('off')

    plt.show()


def butter_worth_low_mask(shape, rank, radius):
    '''
    H(u, v) = 1 / (1 + (D(u, v) / radius)^(2 * rank))
    :param shape:
    :param rank:
    :param radius:
    :return:
    '''
    h, w = shape[:2]
    cx, cy = int(w / 2), int(h / 2)
    # 计算以中心为原点坐标分量
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    # 每个点到中心的距离
    dis = np.sqrt(u * u + v * v)
    filt = 1 / (1 + np.power(dis / radius, 2 * rank))
    return filt

def butter_worth_high_mask(shape, rank, radius):
    '''
    H(u, v) = 1 / (1 + (D(u, v) / radius)^(2 * rank))
    :param shape:
    :param rank:
    :param radius:
    :return:
    '''
    h, w = shape[:2]
    cx, cy = int(w / 2), int(h / 2)
    # 计算以中心为原点坐标分量
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    # 每个点到中心的距离
    dis = np.sqrt(u * u + v * v)
    filt = 1.0 - 1 / (1 + np.power(dis / radius, 2 * rank))
    return filt

def fft_butter_worth_low_pass_test():
    fft_butter_worth_test(butter_worth_low_mask)

def fft_butter_worth_hight_pass_test():
    fft_butter_worth_test(butter_worth_high_mask)

def fft_butter_worth_test(mask_func):
    img = cv2.imread('images/test.png', cv2.IMREAD_GRAYSCALE)
    fft = np.fft.fft2(img)
    center_fft = np.fft.fftshift(fft)
    center_fft_spec = np.log(np.abs(center_fft))

    plt.figure(figsize=(10, 8), dpi=120)
    plt.subplot(341)
    plt.imshow(img, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    plt.subplot(342)
    plt.imshow(center_fft_spec, cmap='gray')
    plt.title('center spec')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 2, 25)
    plt.subplot(343)
    plt.imshow(bw_mask, cmap='gray')
    plt.title('Rk2 Rd25 mask')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 20, 25)
    plt.subplot(344)
    plt.imshow(bw_mask, cmap='gray')
    plt.title('Rk20 Rd25 mask')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 2, 5)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs( np.fft.ifft2(filter_rr) )
    plt.subplot(345)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk2 Rd5 img')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 2, 15)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs( np.fft.ifft2(filter_rr) )
    plt.subplot(346)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk2 Rd15 img')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 2, 30)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs(np.fft.ifft2(filter_rr))
    plt.subplot(347)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk2 Rd30 img')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 2, 80)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs(np.fft.ifft2(filter_rr))
    plt.subplot(348)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk2 Rd80 img')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 20, 5)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs(np.fft.ifft2(filter_rr))
    plt.subplot(349)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk20 Rd5 img')
    plt.axis('off')

    bw_mask = mask_func(img.shape, 20, 15)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs(np.fft.ifft2(filter_rr))
    plt.subplot2grid((3,4), (2, 1),colspan=1,rowspan=1)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk20 Rd15 img')
    plt.axis('off')
    #
    bw_mask = mask_func(img.shape, 2, 30)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs(np.fft.ifft2(filter_rr))
    plt.subplot2grid((3,4), (2, 2),colspan=1,rowspan=1)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk20 Rd30 img')
    plt.axis('off')
    #
    bw_mask = mask_func(img.shape, 2, 80)
    filter_rr = bw_mask * center_fft
    filter_rr_img = np.abs(np.fft.ifft2(filter_rr))
    plt.subplot2grid((3,4), (2, 3),colspan=1,rowspan=1)
    plt.imshow(filter_rr_img, cmap='gray')
    plt.title('Rk20 Rd80 img')
    plt.axis('off')

    plt.show()


def gaussian_low_mask_func(shape, sigma):
    '''
    H(u, v) = e^(-(u^2 + v^2) / (2 * sigma^2))
    :param shape:
    :param sigma:
    :return:
    '''
    h, w = shape[: 2]
    cx, cy = int(w / 2), int(h / 2)
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    dis2 = u * u + v * v
    p = -dis2 / (2 * sigma**2)
    #filt = 1 / (2 * pi * sigma**2) * np.exp(p)
    filt = np.exp(p)
    return filt

def gaussian_high_mask_func(shape, sigma):
    '''
    H(u, v) = e^(-(u^2 + v^2) / (2 * sigma^2))
    :param shape:
    :param sigma:
    :return:
    '''
    h, w = shape[: 2]
    cx, cy = int(w / 2), int(h / 2)
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    dis2 = u * u + v * v
    p = -dis2 / (2 * sigma**2)
    #filt = 1 / (2 * pi * sigma**2) * np.exp(p)
    filt = 1.0 - np.exp(p)
    return filt

def gaussian_low_pass_test():
    gaussian_filter_test(gaussian_low_mask_func)

def gaussian_high_pass_test():
    gaussian_filter_test(gaussian_high_mask_func)

def gaussian_filter_test(mask_func):
    img = cv2.imread('images/test.png', cv2.IMREAD_GRAYSCALE)
    fft = np.fft.fft2(img)
    center_fft = np.fft.fftshift(fft)
    center_fft_spec = np.log(np.abs(center_fft))

    plt.figure(figsize=(8, 4), dpi=150)
    plt.subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(center_fft_spec, cmap='gray')
    plt.title('center spec')
    plt.axis('off')

    g_mask = mask_func(img.shape, 10)
    plt.subplot(243)
    plt.imshow(g_mask, cmap='gray')
    plt.title('D10 mask')
    plt.axis('off')

    g_mask = mask_func(img.shape, 30)
    plt.subplot(244)
    plt.imshow(g_mask, cmap='gray')
    plt.title('D30 mask spec')
    plt.axis('off')

    g_mask = mask_func(img.shape, 5)
    filter_fft = center_fft * g_mask
    filter_fft_img = np.abs( np.fft.ifft2(filter_fft) )
    plt.subplot(245)
    plt.imshow(filter_fft_img, cmap='gray')
    plt.title('D5 img')
    plt.axis('off')

    g_mask = mask_func(img.shape, 10)
    filter_fft = center_fft * g_mask
    filter_fft_img = np.abs(np.fft.ifft2(filter_fft))
    plt.subplot(246)
    plt.imshow(filter_fft_img, cmap='gray')
    plt.title('D10 img')
    plt.axis('off')

    g_mask = mask_func(img.shape, 15)
    filter_fft = center_fft * g_mask
    filter_fft_img = np.abs(np.fft.ifft2(filter_fft))
    plt.subplot(247)
    plt.imshow(filter_fft_img, cmap='gray')
    plt.title('D15 img')
    plt.axis('off')

    g_mask = mask_func(img.shape, 30)
    filter_fft = center_fft * g_mask
    filter_fft_img = np.abs(np.fft.ifft2(filter_fft))
    plt.subplot(248)
    plt.imshow(filter_fft_img, cmap='gray')
    plt.title('D30 img')
    plt.axis('off')

    plt.show()





