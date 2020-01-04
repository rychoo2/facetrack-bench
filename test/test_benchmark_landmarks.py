import shutil
from unittest import TestCase
from pipeline.step2_benchmark_landmarks import benchmark_landmarks_for_datasets
import os
import time

class TestProcess_landmarks(TestCase):
    test_output_path = "tmp/landmarks_benchmark/1"
    expected_output_path = "expected/landmarks_benchmark/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    def test_process_landmarks(self):
        start = time.process_time()
        benchmark_landmarks_for_datasets("data", self.test_output_path)
        duration = time.process_time() - start

        # should have 2 datasets
        self.assertTrue(set(os.listdir(self.test_output_path)).issuperset({'capture0', 'capture1'}))

        # should have expected landmark.csv's
        for dataset in ['capture0', 'capture1']:
            self.assertListEqual(
                self.readfile("{}/{}/landmarks_benchmark.csv".format(self.test_output_path, dataset)),
                self.readfile("{}/{}/landmarks_benchmark.csv".format(self.expected_output_path, dataset))
            )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 130)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

