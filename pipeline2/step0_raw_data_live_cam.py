import os
import cv2
from pynput import mouse
from libs.utils import get_timestamp
from tkinter import Tk

root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

cam = cv2.VideoCapture(0)
mouse_ctrl = mouse.Controller()
now = get_timestamp()

output = '../train_data2/raw/{}'.format(now)
os.makedirs(output+'/images')
fw = open('{}/positions.csv'.format(output), 'w')
fw.write("frame,timestamp,x,y,screen_width,screen_height,image_path,image_height,image_width\n")

def on_move(x, y):
    print('Pointer moved to {0}'.format(
        (x, y)))

def on_click(x, y, button, pressed):
    timestamp = get_timestamp()
    frame = 0
    if pressed:
        print('{0} at {1} at {2}, on screen ({3},{4})'.format(
            'Pressed' if pressed else 'Released',
            (x, y), timestamp, screen_width, screen_height))
        if pressed:
            frame += 1
            ret_val, img = cam.read()
            img_height, img_width = img.shape[0:2]
            image_path = "images/{}_{:03d}.jpg".format(timestamp, frame)
            fw.write("{},{},{},{},{},{},{}\n".format(frame, timestamp, x, y, screen_width, screen_height, image_path, img_height, img_width))
            fw.flush()
            cv2.imwrite("{}/{}".format(output, image_path), img)

if __name__ == '__main__':
    with mouse.Listener(
            on_move=on_move,
           on_click=on_click) as listener:
       listener.join()