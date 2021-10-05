import cv2
import os


def calculate_screen_position(x, y, img_width, img_height):
    return int((1 - x) * img_width), int(y * img_height)


def draw_marker(img, x, y, number, color):
    colors = {"orange": (10, 121, 255), "green": (0, 252, 124), "red": (0, 0, 255)}
    c = colors[color]

    if number is None:
        r1 = 3
        r2 = 18
        number = ''
    else:
        r1 = 2
        r2 = 15
        number = str(number)
    cv2.circle(img, (x, y), r1, c, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r2, c, 2, cv2.LINE_AA)
    cv2.putText(img, number, (x + 3, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .4,
                c, 1, cv2.LINE_AA)


def draw_markers_on_img(marker):
    img = cv2.imread(marker["img_path"])
    img_height, img_width = img.shape[:2]
    for m in marker["markers"]:
        x, y = calculate_screen_position(
            m["x"], m["y"], img_width, img_height)
        number = m["number"]
        color = m["color"]
        draw_marker(img, x, y, number, color)
    return img


def place_markers_on_images(input_path, data, output_path):
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    for entry in data:
        img_dir = "{}/{}/{}/{}".format(input_path, output_path, entry["img_path"].split("/")[-3], "images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_file = entry["img_path"].split('/')[-1]
        img = draw_markers_on_img(entry)
        cv2.imwrite(f"{img_dir}/{img_file}", img)
        cv2.destroyAllWindows()
