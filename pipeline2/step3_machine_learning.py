import pandas as pd
import numpy as np
import os
import collections

import sklearn
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

from libs.utils import get_latest_features, get_timestamp
from pipeline.models import CenterOfScreenModel #, NNSequentialKerasBasic, NNSequentialKerasBasic0
# LinearRegressionBasic, \
# LinearRidgeBasic, LinearLassoBasic, LinearElasticNetBasic, SklearnCustom, PLSRegression, BaggingRegressor, \
# ExtraTreesRegressor, RandomForestRegressorBasic, MultiTaskLassoCV, MLPRegressor, DecisionTreeRegressor, \
# ExtraTreeRegressor

from pipeline.models.sklearn_model_base import SklrearnModelBase

pd.options.display.float_format = "{:.4f}".format

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data2"

models = [
    # [CenterOfScreenModel],
    [SklrearnModelBase, ensemble.BaggingRegressor],
    [SklrearnModelBase, ensemble.RandomForestRegressor],
    [SklrearnModelBase, tree.DecisionTreeRegressor],
    [SklrearnModelBase, tree.ExtraTreeRegressor],
    [SklrearnModelBase, ensemble.ExtraTreesRegressor],
]

training_columns = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                    'gaze_angle_x','gaze_angle_y',
                    'pose_Tx', 'pose_Ty', 'pose_Tz',]
target_columns = ['rel_target_x', 'rel_target_y']

def generate_predictions_for_datasets(input_path, output_path):
    benchmark_result = []
    predictions_df = pd.DataFrame()
    os.makedirs(output_path)
    overall = pd.DataFrame()
    overall_groups = collections.defaultdict(pd.DataFrame)

    features_path, datasets = get_latest_features(input_path)

    dataset_groups = get_dataset_groups(datasets)

    for dataset in datasets:
        dataset_features_path = "{}/{}/features.csv".format(features_path, dataset)
        data = pd.read_csv(dataset_features_path)
        overall = overall.append(data, ignore_index=True)
        overall_groups[dataset_groups[dataset]] = overall_groups[dataset_groups[dataset]].append(data, ignore_index=True)

        benchmark, predictions = generate_predictions(dataset, dataset_features_path, data)
        predictions_df = predictions_df.append(predictions)
        benchmark_result += benchmark

    for grp, df in overall_groups.items():
        benchmark, predictions =  generate_predictions('overall_{}'.format(grp), features_path, df)
        predictions_df = predictions_df.append(predictions)
        benchmark_result += benchmark

    benchmarks_df = pd.DataFrame(benchmark_result)

    predictions_df.to_csv(
        "{}/models_predictions.csv".format(output_path),
        index=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(benchmarks_df.drop(benchmarks_df.columns[0], axis=1))
        benchmarks_df.to_csv(
        "{}/models_benchmark.csv".format(output_path),
            header=['filename', 'dataset', 'model', 'count', 'train_score', 'test_score'],
            index=False)



    return benchmark_result


def get_dataset_groups(datasets):
    d = dict()

    for group, name in [(dataset[dataset.rfind('_')+1:], dataset) for dataset in datasets]:
        d[name] = group

    return d

def generate_predictions(dataset_name, filename, df):
    result = []
    predictions = pd.DataFrame()
    train_input, train_target, test_input, test_target = prepare_dataframe(df)
    train_input_left, train_target_left, test_input_left, test_target_left = prepare_dataframe(df, 'left')
    train_input_right, train_target_right, test_input_right, test_target_right = prepare_dataframe(df, 'right')

    for modelClass, *args in models:
        print("processing dataset:{} model: ".format(dataset_name), modelClass, *args)
        model = modelClass(*args)
        model.train(train_input, train_target)
        train_output = model.predict(train_input)
        test_output = model.predict(test_input)


        train_benchmark = model.evaluate(train_output, train_target)
        test_benchmark = model.evaluate(test_output, test_target)
        result.append([os.path.relpath(filename), dataset_name,
                       model.name, len(train_input.index), train_benchmark, test_benchmark])

        model_left = modelClass(*args)
        model_left.train(train_input_left, train_target_left)
        train_output_left = model_left.predict(train_input_left)
        test_output_left = model_left.predict(test_input_left)
        train_benchmark_left = model_left.evaluate(train_output_left, train_target_left)
        test_benchmark_left = model_left.evaluate(test_output_left, test_target_left)
        result.append([os.path.relpath(filename), dataset_name,
                       model_left.name + '_left', len(train_input_left.index), train_benchmark_left, test_benchmark_left])


        model_right = modelClass(*args)
        model_right.train(train_input_right, train_target_right)
        train_output_right = model_right.predict(train_input_right)
        test_output_right = model_right.predict(test_input_right)
        train_benchmark_right = model_right.evaluate(train_output_right, train_target_right)
        test_benchmark_right = model_right.evaluate(test_output_right, test_target_right)
        result.append([os.path.relpath(filename), dataset_name,
                       model_right.name + '_right', len(train_input_right.index), train_benchmark_right, test_benchmark_right])

        train_output_merged = get_merged_eyes_dataframe(train_input_right, train_output_right, train_input_left, train_output_left)
        test_output_merged = get_merged_eyes_dataframe(test_input_right, test_output_right, test_input_left, test_output_left)
        train_target_merged = train_target_right.join(train_target_left,
                              lsuffix='_l',
                              rsuffix='_r',
                              how='outer')
        test_target_merged = test_target_right.join(test_target_left,
                              lsuffix='_l',
                              rsuffix='_r',
                              how='outer')
        train_target_merged['rel_target_x'] =  train_target_merged[['rel_target_x_r', 'rel_target_x_l']].mean(axis=1)
        train_target_merged['rel_target_y'] =  train_target_merged[['rel_target_y_r', 'rel_target_y_r']].mean(axis=1)
        test_target_merged['rel_target_x'] =  test_target_merged[['rel_target_x_r', 'rel_target_x_l']].mean(axis=1)
        test_target_merged['rel_target_y'] =  test_target_merged[['rel_target_y_r', 'rel_target_y_r']].mean(axis=1)

        train_output_with_target = train_output_merged[['x_avg', 'y_avg']].join(train_target_merged[['rel_target_x', 'rel_target_y']]).dropna()
        test_output_with_target = test_output_merged[['x_avg', 'y_avg']].join(test_target_merged[['rel_target_x', 'rel_target_y']]).dropna()
        train_benchmark_merged = model_right.evaluate(train_output_with_target[['x_avg', 'y_avg']].to_numpy(), train_output_with_target[['rel_target_x', 'rel_target_y']])
        test_benchmark_merged = model_right.evaluate(test_output_with_target[['x_avg', 'y_avg']].to_numpy(), test_output_with_target[['rel_target_x', 'rel_target_y']])
        result.append([os.path.relpath(filename), dataset_name,
                       model_right.name + '_eye_avg', len(train_output_merged.index), train_benchmark_merged,
                       test_benchmark_merged])

        predictions_df = get_prediction_results(df, test_input, test_target, test_output, train_input, train_target,
                                                train_output)

        predictions_df.insert(0, 'dataset', dataset_name)
        predictions_df.insert(1, 'model', model.name)

        predictions = predictions.append(predictions_df)

    return result, predictions

def get_prediction_results(df, test_input, test_target, test_output, train_input, train_target, train_output):
    d1 = test_input.copy()
    d1[['prediction_x', 'prediction_y']] = test_output
    d1[['target_x', 'target_y']] = test_target
    d1['type'] = 'test'
    d2 = train_input.copy()
    d2[['prediction_x', 'prediction_y']] = train_output
    d2[['target_x', 'target_y']] = train_target
    d2['type'] = 'train'
    output_columns = ['frame', 'image_path', 'type', 'prediction_x', 'prediction_y', 'target_x', 'target_y']
    output_df= df.merge(d1.append(d2))
    return output_df[output_columns]

def normalize_dims(ndarray):
    if len(np.shape(ndarray))>1:
        return ndarray
    else:
        return np.expand_dims(ndarray, axis=1)

def prepare_dataframe(df: pd.DataFrame, eye = None):
    res_df = df.copy()
    if eye == 'left':
        excluded_eye = '_0'
    elif eye == 'right':
        excluded_eye = '_1'
    else:
        excluded_eye = None
    if excluded_eye:
        res_df = res_df[[c for c in res_df.columns if excluded_eye not in c]]
    res_df.dropna(inplace=True)
    train, test = train_test_split(res_df, test_size=0.2, random_state=0)

    return (train[[c for c in res_df.columns if c in training_columns]], train[target_columns],
            test[[c for c in res_df.columns if c in training_columns]], test[target_columns])

def get_merged_eyes_dataframe(input_right, output_right, input_left, output_left):
    right = input_right.copy()
    right[['x', 'y']] = pd.DataFrame(output_right, index=right.index)
    left = input_left.copy()
    left[['x', 'y']] = pd.DataFrame(output_left, index=left.index)

    input_merged = right.join(left,
                              lsuffix='_l',
                              rsuffix='_r',
                              how='outer')

    input_merged['x_avg'] = input_merged[['x_l', 'x_r']].mean(axis=1)
    input_merged['y_avg'] = input_merged[['y_l', 'y_r']].mean(axis=1)
    return input_merged


if __name__ == '__main__':
    now = get_timestamp()
    output_dir = "{}/machine_learning/{}".format(train_data_dir, now)
    generate_predictions_for_datasets(train_data_dir, output_dir)
