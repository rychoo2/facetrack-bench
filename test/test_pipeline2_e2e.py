import shutil
from unittest import TestCase
from pipeline2.step3_machine_learning import generate_predictions_for_datasets
from pipeline2.step2_features import generate_features_for_datasets
import time
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import os

class TestPrediction_models(TestCase):
    test_output_path = "tmp/pipeline2/e2e"
    test_input_path = "data/pipeline2/e2e"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)


    def test_accuracy_for_100samples_dataset(self):
        start = time.process_time()

        generate_features_for_datasets(self.test_input_path, f'{self.test_output_path}/features/1')
        benchmark_result = generate_predictions_for_datasets(self.test_output_path, self.test_output_path)

        duration = time.process_time() - start

        model_name, error = benchmark_result


        print(f"Best model {model_name},"
              f"  error: {error}")
        self.assertLess(error, 0.07)

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 150)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()
