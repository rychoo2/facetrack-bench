import sys
import os
import dlib
import glob
import cv2
import numpy as np
from pynput import mouse
import datetime

cam = cv2.VideoCapture(0)
mouse_ctrl = mouse.Controller()
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

output = 'train_data/raw/{}'.format(now)
os.makedirs(output+'/images')
fw = open('{}/positions.csv'.format(output), 'w')

def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

def on_click(x, y, button, pressed):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print('{0} at {1} at {2}'.format(
        'Pressed' if pressed else 'Released',
        (x, y), timestamp))
    ret_val, img = cam.read()
    image_path = "images/{}.jpg".format(timestamp)
    fw.write("{},{},{}\n".format(x, y, image_path))
    fw.flush()
    cv2.imwrite("{}/{}".format(output, image_path), img)

with mouse.Listener(
        on_move=on_move,
        on_click=on_click) as listener:
    listener.join()