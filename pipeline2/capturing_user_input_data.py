import tkinter as tk
import os
import cv2
from libs.utils import get_timestamp, points


class Capture:
    def __init__(self, master, n, output_path):
        self.master = master
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        self.master.state('zoomed')
        self.master.attributes('-fullscreen', True)
        self.master.bind('<Escape>', self.close)
        self.canvas = tk.Canvas(master, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.points = points(n, self.screen_width, self.screen_height)
        self.counter = tk.IntVar(value=0)
        self.counter.trace("w", self.update_point)
        self.radius = 2.5
        self.point_color = "yellow"
        self.info_text = self.canvas.create_text(
            self.screen_width / 2, 10, fill="white", font="Courier 10")
        self.cam = cv2.VideoCapture(0)
        self.output = '{}/raw/{}'.format(output_path, get_timestamp())
        os.makedirs(self.output + '/images')
        self.fw = open('{}/positions.csv'.format(self.output), 'w')
        self.fw.write(
            "frame,timestamp,x,y,screen_width,screen_height,image_path,image_width,image_height\n")
        self.update_point()

    def capture_frame(self, event):
        frame = self.counter.get() + 1
        timestamp = get_timestamp()
        x, y = event.x, event.y
        ret_val, img = self.cam.read()
        img_height, img_width = img.shape[0:2]
        image_path = "images/{}_{:03d}.jpg".format(timestamp, frame)
        self.fw.write("{},{},{},{},{},{},{},{},{}\n".format(frame, timestamp,
                                                            x, y, self.screen_width, self.screen_height, image_path,
                                                            img_width,
                                                            img_height))
        self.fw.flush()
        cv2.imwrite("{}/{}".format(self.output, image_path), img)

    def circle(self, x, y, radius, color="yellow"):
        x0 = x - radius
        y0 = y - radius
        x1 = x + radius
        y1 = y + radius
        return self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline=color, tags="circle")

    def increment_counter(self):
        if self.counter.get() < len(self.points) - 1:
            self.counter.set(self.counter.get() + 1)
        else:
            self.close()

    def update_text(self):
        remaining_dots = len(self.points) - self.counter.get()
        self.canvas.itemconfig(self.info_text,
                               text=f"Click on the yellow dots ({remaining_dots} out of {len(self.points)} dots remaining)")

    def update_point(self, *args):
        self.canvas.delete("circle")
        self.update_text()
        point_x, point_y = self.points[self.counter.get()]
        self.circle(point_x, point_y, self.radius, self.point_color)
        self.canvas.tag_bind("circle", "<1>", self.capture_frame)

        self.increment_counter()

    def close(self, *args):
        self.cam.release()
        self.canvas.delete(self.info_text)
        self.master.destroy()


if __name__ == "__main__":
    n = 3
    output = '../train_data2'
    root = tk.Tk()
    capture = Capture(root, n, output)
    root.mainloop()
