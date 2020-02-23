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
    #NNSequentialKerasBasic(), NNSequentialKerasBasic0(),
    LinearRegressionBasic,
    LinearRidgeBasic,
    LinearLassoBasic,
    LinearElasticNetBasic,
    PLSRegression,
    BaggingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressorBasic,
    MultiTaskLassoCV,
    MLPRegressor,
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
    train_data = {
        None: prepare_dataframe(df),
        'x': prepare_dataframe(df, 'x'),
        'y': prepare_dataframe(df, 'y')}
    for modelClass in models:
        for axis in [None, 'x', 'y']:
            train_input, train_target, test_input, test_target = train_data[axis]
            model = modelClass()
            model.train(train_input, train_target)
            train_output = model.predict(train_input)
            test_output = model.predict(test_input)
            train_benchmark = model.evaluate(train_output, train_target)
            test_benchmark = model.evaluate(test_output, test_target)
            result.append([os.path.relpath(filename), dataset_name if not axis else '{}_{}'.format(dataset_name, axis),
                           model.name, len(df.index), train_benchmark, test_benchmark])
    return result

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
