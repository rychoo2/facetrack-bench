import pandas as pd
import pprint
import os

from libs.utils import get_latest_features, get_timestamp
from pipeline.models import CenterOfScreenModel, NNSequentialKerasBasic, NNSequentialKerasBasic0, LinearRegressionBasic

pp = pprint.PrettyPrinter(indent=4)

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

models = [CenterOfScreenModel(), NNSequentialKerasBasic(), NNSequentialKerasBasic0(), LinearRegressionBasic()]


def benchmark_models(input_path, output_path):
    result = []
    os.makedirs(output_path)
    overall_input = pd.DataFrame()
    overall_target = pd.DataFrame()

    for dataset, path in get_latest_features(input_path):
        data = pd.read_csv("{}/features.csv".format(path))
        data.dropna(subset=['rel_target_x', 'rel_target_y'], inplace=True)
        data.fillna(0.5, inplace=True)

        input = data[['rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y', 'rel_pose_x', 'rel_pose_y',
                      'rel_eye_distance_x', 'rel_eye_distance_y', 'rel_left_pupil_x', 'rel_left_pupil_y',
                      'rel_right_pupil_x', 'rel_right_pupil_y']]
        target = data[['rel_target_x', 'rel_target_y']]

        overall_input = overall_input.append(input)
        overall_target = overall_target.append(target)

        for model in models:
            model.train(input, target)
            output = model.predict(input)
            benchmark = model.evaluate(output, target)
            result.append([dataset, model.name, benchmark])

    for model in models:
        model.train(overall_input, overall_target)
        output = model.predict(overall_input)
        benchmark = model.evaluate(output, overall_target)
        result.append(['overall', model.name, benchmark])

    pd.DataFrame(result).to_csv("{}/benchmark.csv".format(output_path),
                                header=['dataset', 'model', 'score'],
                                index=False)

    pp.pprint(result)
    return result


if __name__ == '__main__':
    now = get_timestamp()
    output_dir = "{}/benchmark/{}".format(train_data_dir, now)
    benchmark_models(train_data_dir, output_dir)
