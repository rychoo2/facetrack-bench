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
             'capture2_landmark_avg', 'capture1_landmark_avg', 'overall_avg', 'capture0_landmark_avg',
             'capture2_landmark_avg_x',
             'capture1_opencv_y',
             'capture1_landmark_avg_x',
             'overall_avg_x',
             'capture2_opencv_y',
             'capture0_dlib_x',
             'capture1_opencv_x',
             'overall_opencv_y',
             'overall_dlib_y',
             'overall_dlib_x',
             'capture2_opencv_x',
             'overall_opencv_x',
             'capture0_landmark_avg_x',
             'capture2_dlib_x',
             'capture1_dlib_y',
             'capture1_dlib_x',
             'capture1_landmark_avg_y',
             'capture0_dlib_y',
             'capture2_dlib_y',
             'capture2_landmark_avg_y',
             'capture0_opencv_y',
             'overall_avg_y',
             'capture0_opencv_x',
             'capture0_landmark_avg_y',
             'overall_opencv_xy',
             'capture1_opencv_xy',
             'capture0_dlib_xy',
             'overall_dlib_xy',
             'capture2_landmark_avg_xy',
             'capture2_opencv_xy',
             'capture0_opencv_xy',
             'capture1_dlib_xy',
             'overall_avg_xy',
             'capture2_dlib_xy',
             'capture1_landmark_avg_xy',
             'capture0_landmark_avg_xy',
             'capture1_landmark_avg_xy2',
             'capture2_dlib_xy2',
             'capture2_opencv_y2',
             'capture2_landmark_avg_y2',
             'capture2_dlib_y2',
             'capture1_opencv_xy2',
             'capture0_dlib_y2',
             'capture0_opencv_y2',
             'capture2_landmark_avg_xy2',
             'capture2_opencv_xy2',
             'capture0_landmark_avg_xy2',
             'capture1_dlib_y2',
             'overall_opencv_y2',
             'capture0_dlib_xy2',
             'capture1_landmark_avg_y2',
             'capture1_dlib_xy2',
             'capture0_landmark_avg_y2',
             'overall_dlib_xy2',
             'overall_opencv_xy2',
             'capture0_opencv_xy2',
             'overall_avg_xy2',
             'overall_avg_y2',
             'overall_dlib_y2',
             'capture1_opencv_y2'
             }
        )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 150)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

