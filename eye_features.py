import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)


def erode(img):
    return cv2.erode(img, kernel, iterations=1)


def dilate(img):
    return cv2.dilate(img, kernel, iterations=1)

def stack(vstack):
    return cv2.pyrDown(np.hstack((vstack,
                      tuple([erode(i) for i in vstack]),
                      tuple([dilate(i) for i in vstack]),
                      tuple([cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel) for i in vstack]),
                      tuple([cv2.morphologyEx(i, cv2.MORPH_CLOSE, kernel) for i in vstack]),
                      tuple([cv2.morphologyEx(i, cv2.MORPH_CROSS, kernel) for i in vstack]),
                      tuple([cv2.morphologyEx(i, cv2.MORPH_HITMISS, kernel) for i in vstack]),
                      tuple([cv2.morphologyEx(i, cv2.MORPH_RECT, kernel) for i in vstack]),
                      tuple([cv2.Canny(i, 0, 50) for i in vstack]),
                      tuple([cv2.Canny(i, 50, 100) for i in vstack]),
                      tuple([cv2.Canny(i, 100, 150) for i in vstack]),
                      tuple([cv2.Canny(i, 150, 250) for i in vstack]),
                      )))


for image_file in ['./images/eye3.png', './images/eye4.png']:
    eyeimg = cv2.imread(image_file)
    eyeimg2 = cv2.GaussianBlur(eyeimg, (5,5), 0)
    eyeimg3 = cv2.cvtColor(eyeimg2, cv2.COLOR_BGR2GRAY)
    eyeimg3 = cv2.equalizeHist(eyeimg3)
    eyeimg_final = cv2.cvtColor(eyeimg3, cv2.COLOR_BGR2RGB)
    # eyeimg_final = cv2.morphologyEx(eyeimg_final, cv2.MORPH_OPEN, kernel)
    cv2.imshow(image_file,
               stack(np.vstack((
                   eyeimg,
                   eyeimg_final,
                   cv2.cvtColor(cv2.threshold(eyeimg_final, 150, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.threshold(eyeimg_final, 50, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.threshold(eyeimg_final, 0, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 5, 0), cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 5, 1), cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 5, 2), cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 7, 2), cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 9, 2), cv2.COLOR_BGR2RGB),
                   cv2.cvtColor(cv2.adaptiveThreshold(eyeimg3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                      cv2.THRESH_BINARY, 11, 2), cv2.COLOR_BGR2RGB),
                   ))))

cv2.waitKey(0)
