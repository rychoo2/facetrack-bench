import sys
import os
import dlib
import glob
import cv2
import numpy as np
from pynput import mouse
import datetime
from libs.utils import get_timestamp
from tkinter import Tk

root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

cam = cv2.VideoCapture(0)
mouse_ctrl = mouse.Controller()
now = get_timestamp()

output = 'train_data/raw/{}'.format(now)
os.makedirs(output+'/images')
fw = open('{}/positions.csv'.format(output), 'w')

def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

def on_click(x, y, button, pressed):
    timestamp = get_timestamp()
    if pressed:
        print('{0} at {1} at {2}, on screen ({3},{4})'.format(
            'Pressed' if pressed else 'Released',
            (x, y), timestamp, screen_width, screen_height))
        if pressed:
            ret_val, img = cam.read()
            image_path = "images/{}.jpg".format(timestamp)
            fw.write("{},{},{},{},{},{}\n".format(timestamp, x, y, screen_width, screen_height, image_path))
            fw.flush()
            cv2.imwrite("{}/{}".format(output, image_path), img)

if __name__ == '__main__':
    with mouse.Listener(
            on_move=on_move,
           on_click=on_click) as listener:
       listener.join()