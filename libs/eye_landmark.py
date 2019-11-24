import cv2
detector = cv2.SimpleBlobDetector_create()

def detect_pupil(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.equalizeHist(img)
    img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    keypoints = detector.detect(img)
    if len(keypoints) > 0:
        keypoints.sort(key=lambda s: s.size, reverse=True)
        return keypoints[0]
    else:
        return None
