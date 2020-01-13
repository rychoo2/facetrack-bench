import pandas as pd
import numpy as np
import os
import collections

from sklearn.model_selection import train_test_split

from libs.utils import get_latest_features, get_timestamp
from pipeline.models import CenterOfScreenModel, NNSequentialKerasBasic, NNSequentialKerasBasic0, LinearRegressionBasic, \
    LinearRidgeBasic, LinearLassoBasic, LinearElasticNetBasic, SklearnCustom, PLSRegression, BaggingRegressor, \
    ExtraTreesRegressor, RandomForestRegressorBasic, MultiTaskLassoCV, MLPRegressor, DecisionTreeRegressor, \
    ExtraTreeRegressor

pd.options.display.float_format = "{:.4f}".format

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

models = [
    CenterOfScreenModel(),
    #NNSequentialKerasBasic(), NNSequentialKerasBasic0(),
    LinearRegressionBasic(),
    LinearRidgeBasic(),
    LinearLassoBasic(),
    LinearElasticNetBasic(),
    PLSRegression(),
    BaggingRegressor(),
    ExtraTreesRegressor(),
    RandomForestRegressorBasic(),
    MultiTaskLassoCV(),
    MLPRegressor(),
    DecisionTreeRegressor(),
    ExtraTreeRegressor(),
    SklearnCustom(),
]

training_columns = ['rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y', 'rel_pose_x', 'rel_pose_y',
                      'rel_eye_distance_x', 'rel_eye_distance_y', 'rel_left_pupil_x', 'rel_left_pupil_y',
                      'rel_right_pupil_x', 'rel_right_pupil_y']

target_columns = ['rel_target_x', 'rel_target_y']


def benchmark_models_for_datasets(input_path, output_path):
    result = []
    os.makedirs(output_path)
    overall = pd.DataFrame()
    overall_groups = collections.defaultdict(pd.DataFrame)

    features_path, datasets = get_latest_features(input_path)

    dataset_groups = get_dataset_groups(datasets)

    for dataset in datasets:
        dataset_features_path = "{}/{}/features.csv".format(features_path, dataset)
        data = pd.read_csv(dataset_features_path)
        overall = overall.append(data)
        overall_groups[dataset_groups[dataset]] = overall_groups[dataset_groups[dataset]].append(data)

        result += benchmark_models(dataset, dataset_features_path, data)

    for grp, df in overall_groups.items():
        result += benchmark_models('overall_{}'.format(grp), features_path, df)

    #result += benchmark_models('overall', features_path, overall)

    output_df = pd.DataFrame(result)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(output_df.drop(output_df.columns[0], axis=1))
    output_df.to_csv(
        "{}/models_benchmark.csv".format(output_path),
            header=['filename', 'dataset', 'model', 'count', 'train_score', 'test_score'],
            index=False)

    return result


def get_dataset_groups(datasets):
    d = dict()

    for group, name in [(dataset[dataset.rfind('_')+1:], dataset) for dataset in datasets]:
        d[name] = group

    return d

def benchmark_models(dataset_name, filename, df):
    result = []
    train_x, train_y, test_x, test_y = prepare_dataframe(df)
    for model in models:
        model.train(train_x, train_y)
        train_output = model.predict(train_x)
        test_output = model.predict(test_x)
        train_benchmark = model.evaluate(train_output, train_y)
        test_benchmark = model.evaluate(test_output, test_y)
        result.append([os.path.relpath(filename), dataset_name, model.name, len(df.index), train_benchmark, test_benchmark])
    return result

def prepare_dataframe(df):
    df.dropna(subset=target_columns, inplace=True)
    df.fillna(0.5, inplace=True)
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    return (train[training_columns], train[target_columns],
            test[training_columns], test[target_columns])

if __name__ == '__main__':
    now = get_timestamp()
    output_dir = "{}/models_benchmark/{}".format(train_data_dir, now)
    benchmark_models_for_datasets(train_data_dir, output_dir)
