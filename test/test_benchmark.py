import shutil
from unittest import TestCase
from pipeline.step3_benchmark import benchmark_models_for_datasets
import time

class TestBenchmark_models(TestCase):
    test_output_path = "tmp/benchmark/1"
    expected_output_path = "expected/benchmark/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    def test_generate_benchmarks(self):
        start = time.process_time()
        benchmark_models_for_datasets("data", self.test_output_path)
        duration = time.process_time() - start

        self.assertTrue(
            len(self.readfile("{}/benchmark.csv".format(self.test_output_path))) >= 10,
        )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 130)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

