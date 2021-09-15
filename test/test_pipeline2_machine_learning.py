import shutil
from unittest import TestCase
from pipeline2.step3_machine_learning import generate_predictions_for_datasets
import time
import pandas as pd

class TestPrediction_models(TestCase):
    test_output_path = "tmp/pipeline2/machine_learning/1"
    expected_output_path = "expected/pipeline2/machine_learning/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    # def test_benchmark_models_for_datasets(self):
    #     pass
    #     #


    def test_predictions_output_for_datasets(self):
        start = time.process_time()
        generate_predictions_for_datasets("data/pipeline2", self.test_output_path)
        duration = time.process_time() - start

        output = pd.read_csv("{}/predictions.csv".format(self.test_output_path))
        self.assertTrue(
            len(output) > 70
        )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 150)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

