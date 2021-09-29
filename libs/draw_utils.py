import cv2
import os

colors = {"orange": (10, 121, 255), "green": (0, 252, 124), "red": (0, 0, 255)}


def calculate_screen_position(x, y, img_width, img_height):
    return int((1 - x) * img_width), int(y * img_height)


def draw_marker(img, x, y, model_id, type):
    r1 = 2
    r2 = 15
    if type == "train":
        color = colors["orange"]
    elif type == "test":
        color = colors["red"]
    elif type == 'target':
        r1 = 3
        r2 = 18
        color = colors["green"]
    cv2.circle(img, (x, y), r1, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r2, color, 2, cv2.LINE_AA)
    cv2.putText(img, model_id, (x + 3, y + 4), cv2.FONT_HERSHEY_SIMPLEX, .4,
                color, 1, cv2.LINE_AA)


def draw_markers_on_img(pp):
    img = cv2.imread(pp["file_path"])
    img_height, img_width = img.shape[:2]
    target_x, target_y = pp["target_x"], pp["target_y"]
    draw_marker(img, *calculate_screen_position(target_x, target_y, img_width, img_height), "", "target")
    for m in pp["models"]:
        x, y = calculate_screen_position(
            m["prediction_x"], m["prediction_y"], img_width, img_height)
        model_id = str(m["model_id"])
        pr_type = m["type"]
        draw_marker(img, x, y, model_id, pr_type)
    return img


def place_markers_on_images(data, path):
    output_path = "{}/images".format(path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for entry in data:
        img_dir = "{}/{}".format(output_path,entry["file_path"].split("/")[-3])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_file = entry["file_path"].split('/')[-1]
        img = draw_markers_on_img(entry)
        cv2.imwrite(f"{img_dir}/{img_file}", img)
        cv2.destroyAllWindows()
