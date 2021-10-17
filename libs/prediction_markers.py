import os
import pandas as pd
from .draw_utils import place_markers_on_images

csv_path = "tmp/pipeline2/machine_learning/1/models_predictions.csv"
root_path = "../test"


def prepare_data_from_predictions_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~df['dataset'].str.startswith('overall')]  # drop rows which starts with overall...
    unique_files = df['landmark_image'].unique()  # find unique image filenames
    data = []
    for u_file in unique_files:  # prepare data for drawing function
        u_file_df = df.loc[df["landmark_image"] == u_file]
        # dataset = u_file_df.iloc[0, 0]
        target_x = u_file_df.iloc[0, 9]
        target_y = u_file_df.iloc[0, 10]
        markers = [{"x": target_x, "y": target_y, "number": None, "color": "green"}]
        for row in u_file_df.itertuples():
            if row.type == "train":
                color = "orange"
            elif row.type == "test":
                color = "red"
            markers.append({
                "number": row.model_id,
                "color": color,
                "x": row.prediction_x,
                "y": row.prediction_y
            })
        img_predictions_data = {
            "img_path": f"{u_file}",
            "markers": markers
        }
        data.append(img_predictions_data)
    return data


def draw_prediction_markers(csv_path):
    output_path = os.path.dirname(os.path.relpath(csv_path))
    data = prepare_data_from_predictions_csv(csv_path)
    place_markers_on_images(data, output_path)


if __name__ == '__main__':
    draw_prediction_markers(csv_path)
