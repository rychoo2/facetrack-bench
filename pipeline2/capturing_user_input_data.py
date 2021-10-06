import tkinter as tk
import os
import cv2
import numpy
import math
from libs.utils import get_timestamp

n = 5
radius = 2.5

# A Halton sequence is a sequence of points within a unit dim-cube which
# have low discrepancy (that is, they appear to be randomish but cover
# the domain uniformly)
def halton(dim: int, nbpts: int):
    h = numpy.full(nbpts * dim, numpy.nan)
    p = numpy.full(nbpts, numpy.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)

# Generate list of n points xy coordinates for given screen dimensions
def points(n, screen_w, screen_h):
    hlt = halton(2, n)
    points = (hlt*(screen_w, screen_h)).astype(int)
    return points.tolist()

root = tk.Tk()
root.frame = 0
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# Create full screen canvas
root.state('zoomed')
root.attributes('-fullscreen', True)
canvas = tk.Canvas(root, bg='black', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

pts = points(n, screen_width, screen_height)
points_iterator = iter(pts)

cam = cv2.VideoCapture(0)
now = get_timestamp()
output = './train_data2/raw/{}'.format(now)
os.makedirs(output+'/images')
fw = open('{}/positions.csv'.format(output), 'w')
fw.write("frame,timestamp,x,y,screen_width,screen_height,image_path,image_width,image_height\n")

def close(e):
    cam.release()
    root.destroy()

root.bind('<Escape>', close)

def circle(x, y, radius, canvasName, color="yellow"):
    x0 = x-radius
    y0 = y-radius
    x1 = x+radius
    y1 = y+radius
    return canvasName.create_oval(x0, y0, x1, y1, fill=color, outline=color, tags="circle")


def draw(canvasName):
    try:
        p = next(points_iterator)
        circle(p[0], p[1], radius, canvasName)
        canvas.tag_bind("circle", "<1>", on_click)
    except StopIteration:
        close(StopIteration)


def on_click(event):
    root.frame += 1
    timestamp = get_timestamp()
    x, y = event.x, event.y
    ret_val, img = cam.read()
    img_height, img_width = img.shape[0:2]
    image_path = "images/{}_{:03d}.jpg".format(timestamp, root.frame)
    fw.write("{},{},{},{},{},{},{},{},{}\n".format(root.frame, timestamp,
             x, y, screen_width, screen_height, image_path, img_width, img_height))
    fw.flush()
    cv2.imwrite("{}/{}".format(output, image_path), img)
    canvas.delete("circle")
    draw(canvas)




if __name__ == "__main__":
    draw(canvas)
    root.mainloop()
