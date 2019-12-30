import shutil
from unittest import TestCase
from pipeline.step3_benchmark import benchmark_models
import os
import time

class TestBenchmark_models(TestCase):
    test_output_path = "tmp/benchmark/1"
    expected_output_path = "expected/benchmark/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    def test_generate_benchmarks(self):
        start = time.process_time()
        benchmark_models("data", self.test_output_path)
        duration = time.process_time() - start

        self.assertListEqual(
            self.readfile("{}/benchmark.csv".format(self.test_output_path)),
            self.readfile("{}/benchmark.csv".format(self.expected_output_path))
        )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 130)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

