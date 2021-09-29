import os
import pandas as pd
from draw_utils import place_markers_on_images
from utils import list_dirs

test_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../test"
ml_dir = "{}/tmp/pipeline2/machine_learning".format(test_data_dir)


def get_ml_predictions_dirs(directory):
    ml_dir_sets = list_dirs(directory)
    return ["{}/{}".format(ml_dir, x) for x in ml_dir_sets]


def prepare_data_from_predictions_csv(csv_dir_path):
    df = pd.read_csv("{}/models_predictions.csv".format(csv_dir_path))
    df = df[~df['dataset'].str.startswith('overall')]  # drop rows which starts with overall...
    unique_files = df['landmark_image'].unique()  # find unique image filenames
    data_to_draw = []

    for u_file in unique_files:  # prepare data for drawing function
        u_file_df = df.loc[df["landmark_image"] == u_file]
        file_path = "{}/{}".format(test_data_dir, u_file)
        dataset = u_file_df.iloc[0, 0]
        target_x = u_file_df.iloc[0, 9]
        target_y = u_file_df.iloc[0, 10]
        models = []
        for row in u_file_df.itertuples():
            models.append({
                "model_id": row.model_id,
                "type": row.type,
                "prediction_x": row.prediction_x,
                "prediction_y": row.prediction_y
            })
        img_predictions_data = {
            "dataset": dataset,
            "file_path": file_path,
            "target_x": target_x,
            "target_y": target_y,
            "models": models
        }
        data_to_draw.append(img_predictions_data)
    return data_to_draw


def main():
    for path in get_ml_predictions_dirs(ml_dir):
        place_markers_on_images(prepare_data_from_predictions_csv(path), path)


if __name__ == '__main__':
    main()

