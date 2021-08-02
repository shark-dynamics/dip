import cv2
import numpy as np
import random
import util.noise as noise
import enhance.space_add_mean as se
import enhance.space_gray_map as gm
import enhance.histogram_balance as hb
import enhance.filter as filter
import enhance.freq_low_pass as low_freq
import enhance.freq_high_pass as high_freq
import enhance.img_recover as recover
import enhance.morph as morph
import enhance.scale as scale
import enhance.border as border

if __name__ == '__main__':

    # noise.noise_test()
    # se.add_meat_test()
    # gm.inverse_test()
    # gm.segment_enhance_test()
    # gm.log_enhance_test()
    # gm.r_enhance_test()
    # gm.gray_cut_test()
    # gm.threshold_cut_test()
    # gm.bitmap_cut_test()
    # hb.histogram_test()
    # filter.filter_average_test()
    # filter.filter_weight_avg_test()
    # filter.filter_image_mid_value_test()
    # filter.filter_image_max_min_value_test()
    # filter.filter_middle_value_test()
    # filter.filter_sobel_test()
    # filter.filter_laplace_test()
    # low_freq.fft_test()
    # low_freq.fft_idea_low_pass_test()
    # low_freq.fft_butter_worth_low_pass_test()
    # low_freq.fft_butter_worth_hight_pass_test()
    low_freq.gaussian_low_pass_test()
    # low_freq.gaussian_high_pass_test()
    # high_freq.fft_idea_high_pass_test()
    # recover.motion_blur_recover_test()
    # morph.drode_test()
    # scale.scale_test()
    # border.canny_test()

    cv2.waitKey(0)
