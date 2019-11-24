import random

import cv2
import numpy as np
from libs.eye_landmark import detect_pupil

kernel = np.ones((5,5), np.uint8)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1  # The dot in 20pt font has area of about 30
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByConvexity = False
params.minConvexity = 0.8
params.filterByInertia = True
params.minInertiaRatio = 0.4
detector = cv2.SimpleBlobDetector_create()


def erode(img):
    return cv2.erode(img, kernel, iterations=1)


def dilate(img):
    return cv2.dilate(img, kernel, iterations=1)

def detect_contours(img, flag, accuracy = 3, color = (128, 255, 255), hull = False):
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    vis = np.zeros((h, w, 3), np.uint8)
    contours0, hierarchy = cv2.findContours( img.copy(), cv2.RETR_LIST, flag)
    hulls = [cv2.convexHull(cnt) for cnt in contours0]
    contours = [cv2.approxPolyDP(cnt, accuracy, False) for cnt in contours0]

    if hull:
        cv2.drawContours(vis, hulls, -1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.drawContours(vis, contours, -1, color, 1, cv2.LINE_AA)
    return vis

def detect_circle(img):
    im = img.copy()
    blur = cv2.medianBlur(cv2.cvtColor( img, cv2.COLOR_RGB2GRAY), 3)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 10, 2)
    h, w = img.shape[:2]
    if circles is not None:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(im, (i[0], i[1]), i[2], (255, 0, 0), 2)

            # draw the center of the circle
            cv2.circle(im, (i[0], i[1]), 2, (0, 255, 0), 5)
    return im



def draw_pupil(img):
    im = img.copy()
    keypoint = detect_pupil(img)
    if keypoint:
        print(keypoint.pt)
        cv2.circle(im, (round(keypoint.pt[0]), round(keypoint.pt[1])), 1, (0, 0, 255), 2)
    return im


def detect_blobs(img):
    im = img.copy()
    keypoints = detector.detect(im)
    # print(keypoints)
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(im, keypoints, blank, (0, 255, 255),
                              cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return blobs


def imggrid(vrow):
    imggrid = [
                     vrow,
                      [erode(i) for i in vrow],
                      [dilate(i) for i in vrow],
                      [cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel) for i in vrow],
                      [cv2.morphologyEx(i, cv2.MORPH_CLOSE, kernel) for i in vrow],
                      [cv2.morphologyEx(i, cv2.MORPH_CROSS, kernel) for i in vrow],
                      [cv2.cvtColor(cv2.Canny(i, 50, 250), cv2.COLOR_GRAY2RGB) for i in vrow],
                      ]
    imggrid_3d = []
    for vr in imggrid:
        imggrid_3d.append(vr)
        imggrid_3d.append(detect_contours(i, cv2.CHAIN_APPROX_SIMPLE) for i in vr)
        imggrid_3d.append(detect_contours(i, cv2.CHAIN_APPROX_SIMPLE, hull=True) for i in vr)
        imggrid_3d.append(detect_circle(i) for i in vr)
        imggrid_3d.append(detect_blobs(i) for i in vr)

    return np.hstack((np.vstack(tuple(row)) for row in imggrid_3d))


for image_file in ['./images/eye5.png', './images/eye6.png']:
    eyeimg = cv2.imread(image_file)
    # cv2.circle(eyeimg, (50, 50), 5, (255, 0,0 ), 3)
    eyeimg2 = cv2.GaussianBlur(eyeimg, (5,5), 0)
    eyeimg3 = cv2.cvtColor(eyeimg2, cv2.COLOR_RGB2GRAY)
    eyeimg3 = cv2.equalizeHist(eyeimg3)
    eyeimg_final = cv2.cvtColor(eyeimg3, cv2.COLOR_GRAY2RGB)

    # eyeimg_final = cv2.morphologyEx(eyeimg_final, cv2.MORPH_OPEN, kernel)
    cv2.imshow(image_file,
               cv2.pyrDown(imggrid([
                   eyeimg,
                   eyeimg_final,
                   draw_pupil(eyeimg),
                   # cv2.threshold(eyeimg_final, 150, 255, cv2.THRESH_BINARY)[1],
                   # cv2.threshold(eyeimg_final, 50, 255, cv2.THRESH_BINARY)[1],
                   cv2.threshold(eyeimg_final, 0, 255, cv2.THRESH_BINARY)[1],
                   cv2.threshold(eyeimg_final, 1, 255, cv2.THRESH_BINARY)[1],
                   cv2.threshold(eyeimg_final, 5, 255, cv2.THRESH_BINARY)[1],
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
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 11, 3), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 11, 4), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 13, 1), cv2.COLOR_GRAY2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 13, 2), cv2.COLOR_GRAY2RGB)
                   ])))

cv2.waitKey(0)
