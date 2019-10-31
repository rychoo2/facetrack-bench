import random

import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)


def erode(img):
    return cv2.erode(img, kernel, iterations=1)


def dilate(img):
    return cv2.dilate(img, kernel, iterations=1)

def add_contours(img, flag):
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    contours0, hierarchy = cv2.findContours( img.copy(), cv2.RETR_LIST, flag)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    vis = np.zeros((h, w, 3), np.uint8)
    levels = 3
    cv2.drawContours(vis, contours, (-1, 2)[levels <= 0], (128, 255, 255),
                    3, cv2.LINE_AA, hierarchy, abs(levels))
    return vis

def stack(vstack):
    return np.hstack((
                      np.vstack(tuple(vstack)),
                      np.vstack(tuple([erode(i) for i in vstack])),
                      np.vstack(tuple([dilate(i) for i in vstack])),
                      np.vstack(tuple([cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel) for i in vstack])),
                      np.vstack(tuple([cv2.morphologyEx(i, cv2.MORPH_CLOSE, kernel) for i in vstack])),
                      np.vstack(tuple([cv2.morphologyEx(i, cv2.MORPH_CROSS, kernel) for i in vstack])),
                      np.vstack(tuple([cv2.cvtColor(cv2.Canny(i, 0, 50), cv2.COLOR_GRAY2RGB) for i in vstack])),
                      np.vstack(tuple([cv2.cvtColor(cv2.Canny(i, 50, 100), cv2.COLOR_GRAY2RGB) for i in vstack])),
                      np.vstack(tuple([cv2.cvtColor(cv2.Canny(i, 100, 150), cv2.COLOR_GRAY2RGB) for i in vstack])),
                      np.vstack(tuple([cv2.cvtColor(cv2.Canny(i, 150, 250), cv2.COLOR_GRAY2RGB) for i in vstack])),
                      np.vstack(tuple([add_contours(i, cv2.CHAIN_APPROX_SIMPLE) for i in vstack])),
                      np.vstack(tuple([add_contours(cv2.cvtColor(cv2.Canny(i, 150, 250), cv2.COLOR_GRAY2RGB), cv2.CHAIN_APPROX_SIMPLE) for i in vstack])),
                      ))


for image_file in ['./images/eye3.png', './images/eye4.png']:
    eyeimg = cv2.imread(image_file)
    # cv2.circle(eyeimg, (50, 50), 5, (255, 0,0 ), 3)
    # eyeimg2 = cv2.GaussianBlur(eyeimg, (5,5), 0)
    eyeimg3 = cv2.cvtColor(eyeimg, cv2.COLOR_RGB2GRAY)
    eyeimg3 = cv2.equalizeHist(eyeimg3)
    eyeimg_final = cv2.cvtColor(eyeimg3, cv2.COLOR_GRAY2RGB)

    # eyeimg_final = cv2.morphologyEx(eyeimg_final, cv2.MORPH_OPEN, kernel)
    cv2.imshow(image_file,
               stack([
                   eyeimg,
                   eyeimg_final,
                   cv2.threshold(eyeimg_final, 150, 255, cv2.THRESH_BINARY)[1],
                   cv2.threshold(eyeimg_final, 50, 255, cv2.THRESH_BINARY)[1],
                   cv2.threshold(eyeimg_final, 0, 255, cv2.THRESH_BINARY)[1],
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 5, 0), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 5, 1), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 5, 2), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 7, 2), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 9, 2), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 11, 2), cv2.COLOR_GRAY2RGB),
                   ]))

cv2.waitKey(0)
