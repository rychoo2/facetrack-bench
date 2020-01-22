import shutil
from unittest import TestCase
from pipeline.step4_benchmark_models import benchmark_models_for_datasets
import time
import pandas as pd

class TestBenchmark_models(TestCase):
    test_output_path = "tmp/models_benchmark/1"
    expected_output_path = "expected/models_benchmark/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    def test_benchmark_models_for_datasets(self):
        start = time.process_time()
        benchmark_models_for_datasets("data", self.test_output_path)
        duration = time.process_time() - start

        output = pd.read_csv("{}/models_benchmark.csv".format(self.test_output_path))
        self.assertTrue(
            len(output) > 70
        )
        self.assertEquals(
            set(output.dataset.unique()),
            {'capture0_dlib', 'capture0_opencv', 'capture1_dlib', 'capture1_opencv', 'overall_dlib',
             'overall_opencv', 'capture2_dlib', 'capture2_opencv',
             'capture2_landmark_avg', 'capture1_landmark_avg', 'overall_avg', 'capture0_landmark_avg' }
        )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 130)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

