import cv2
import numpy as np

img = cv2.imread('images/moon.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny_th_func(lt):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lt, lt * 2)
    cv2.imshow('canny', detected_edges)

def canny_test():
    low_th = 10
    max_high_th = 200

    cv2.imshow('origin', img)

    cv2.namedWindow('canny')

    cv2.createTrackbar('low th:', 'canny', low_th, max_high_th, canny_th_func)

    canny_th_func(low_th)