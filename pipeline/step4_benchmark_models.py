import pandas as pd
import numpy as np
import os
import collections

from sklearn.model_selection import train_test_split

from libs.utils import get_latest_features, get_timestamp, training_columns, target_columns
from pipeline.models import CenterOfScreenModel, NNSequentialKerasBasic, NNSequentialKerasBasic0, LinearRegressionBasic, \
    LinearRidgeBasic, LinearLassoBasic, LinearElasticNetBasic, SklearnCustom, PLSRegression, BaggingRegressor, \
    ExtraTreesRegressor, RandomForestRegressorBasic, MultiTaskLassoCV, MLPRegressor, DecisionTreeRegressor, \
    ExtraTreeRegressor

pd.options.display.float_format = "{:.4f}".format

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

models = [
    CenterOfScreenModel,
#    NNSequentialKerasBasic, NNSequentialKerasBasic0,
#     LinearRegressionBasic,
    # LinearRidgeBasic,
    # LinearLassoBasic,
    # LinearElasticNetBasic,
    # PLSRegression,
    BaggingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressorBasic,
    # MultiTaskLassoCV,
    # MLPRegressor,
    DecisionTreeRegressor,
    ExtraTreeRegressor,
    SklearnCustom,
]



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
    train_input, train_target, test_input, test_target = prepare_dataframe(df)
    train_input_x, train_target_x, test_input_x, test_target_x = (d[[c for c in d.columns if c.endswith('x')]]
                                                                  for d in (train_input, train_target, test_input, test_target))
    train_input_y, train_target_y, test_input_y, test_target_y = (d[[c for c in d.columns if c.endswith('y')]]
                                                                  for d in (train_input, train_target, test_input, test_target))

    train_input_x2, train_target_x2, test_input_x2, test_target_x2 = (
        train_input, train_target[[c for c in train_target.columns if c.endswith('x')]],
        test_input, test_target[[c for c in test_target.columns if c.endswith('x')]])

    train_input_y2, train_target_y2, test_input_y2, test_target_y2 = (
        train_input, train_target[[c for c in train_target.columns if c.endswith('y')]],
        test_input, test_target[[c for c in test_target.columns if c.endswith('y')]])

    for modelClass in models:
        model = modelClass()
        model.train(train_input, train_target)
        train_output = model.predict(train_input)
        test_output = model.predict(test_input)
        train_benchmark = model.evaluate(train_output, train_target)
        test_benchmark = model.evaluate(test_output, test_target)
        result.append([os.path.relpath(filename), dataset_name,
                       model.name, len(df.index), train_benchmark, test_benchmark])

        model_x = modelClass()
        model_x.train(train_input_x, train_target_x)
        train_output_x = model_x.predict(train_input_x)
        test_output_x = model_x.predict(test_input_x)
        train_benchmark_x = model_x.evaluate(train_output_x, train_target_x)
        test_benchmark_x = model_x.evaluate(test_output_x, test_target_x)
        result.append([os.path.relpath(filename), dataset_name + '_x',
                       model.name, len(df.index), train_benchmark_x, test_benchmark_x])

        model_y = modelClass()
        model_y.train(train_input_y, train_target_y)
        train_output_y = model_y.predict(train_input_y)
        test_output_y = model_y.predict(test_input_x)
        train_benchmark_y = model_y.evaluate(train_output_y, train_target_y)
        test_benchmark_y = model_y.evaluate(test_output_y, test_target_y)
        result.append([os.path.relpath(filename), dataset_name + '_y',
                       model.name, len(df.index), train_benchmark_y, test_benchmark_y])

        model_x2 = modelClass()
        model_x2.train(train_input_x2, train_target_x2)
        train_output_x2 = model_x2.predict(train_input_x2)
        test_output_x2 = model_x2.predict(test_input_x2)
        train_benchmark_x2 = model_x2.evaluate(train_output_x2, train_target_x2)
        test_benchmark_x2 = model_x2.evaluate(test_output_x2, test_target_x2)
        result.append([os.path.relpath(filename), dataset_name + '_x2',
                       model.name, len(df.index), train_benchmark_x2, test_benchmark_x2])

        model_y2 = modelClass()
        model_y2.train(train_input_y2, train_target_y2)
        train_output_y2 = model_y2.predict(train_input_y2)
        test_output_y2 = model_y2.predict(test_input_y2)
        train_benchmark_y2 = model_y2.evaluate(train_output_y2, train_target_y2)
        test_benchmark_y2 = model_y2.evaluate(test_output_y2, test_target_y2)
        result.append([os.path.relpath(filename), dataset_name + '_y2',
                       model.name, len(df.index), train_benchmark_y2, test_benchmark_y2])



        train_benchmark_xy = model.evaluate(np.concatenate([normalize_dims(train_output_x),
                                                            normalize_dims(train_output_y)], axis=1),
                                            pd.concat([train_target_x, train_target_y], axis=1))
        test_benchmark_xy = model.evaluate(np.concatenate([normalize_dims(test_output_x),
                                                           normalize_dims(test_output_y)], axis=1),
                                           pd.concat([test_target_x, test_target_y], axis=1))
        result.append([os.path.relpath(filename), dataset_name + '_xy',
                       model.name, len(df.index), train_benchmark_xy, test_benchmark_xy])


        train_benchmark_xy2 = model.evaluate(np.concatenate([normalize_dims(train_output_x),
                                                            normalize_dims(train_output_y2)], axis=1),
                                            pd.concat([train_target_x, train_target_y2], axis=1))
        test_benchmark_xy2 = model.evaluate(np.concatenate([normalize_dims(test_output_x),
                                                           normalize_dims(test_output_y2)], axis=1),
                                           pd.concat([test_target_x, test_target_y2], axis=1))

        result.append([os.path.relpath(filename), dataset_name + '_xy2',
                       model.name, len(df.index), train_benchmark_xy2, test_benchmark_xy2])

        train_benchmark_x2y2 = model.evaluate(np.concatenate([normalize_dims(train_output_x2),
                                                              normalize_dims(train_output_y2)], axis=1),
                                              pd.concat([train_target_x2, train_target_y2], axis=1))
        test_benchmark_x2y2 = model.evaluate(np.concatenate([normalize_dims(test_output_x2),
                                                             normalize_dims(test_output_y2)], axis=1),
                                             pd.concat([test_target_x2, test_target_y2], axis=1))

        result.append([os.path.relpath(filename), dataset_name + '_x2y2',
                   model.name, len(df.index), train_benchmark_x2y2, test_benchmark_x2y2])

    return result

def normalize_dims(ndarray):
    if len(np.shape(ndarray))>1:
        return ndarray
    else:
        return np.expand_dims(ndarray, axis=1)

def prepare_dataframe(df, axis=None):
    df.dropna(subset=target_columns, inplace=True)
    df.fillna(0.5, inplace=True)
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    _training_columns = training_columns
    _target_columns = target_columns
    if axis:
        _training_columns = [c for c in _training_columns if c.endswith(axis)]
        _target_columns = [c for c in _target_columns if c.endswith(axis)]
    return (train[_training_columns], train[_target_columns],
            test[_training_columns], test[_target_columns])

if __name__ == '__main__':
    now = get_timestamp()
    output_dir = "{}/models_benchmark/{}".format(train_data_dir, now)
    benchmark_models_for_datasets(train_data_dir, output_dir)
