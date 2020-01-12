import shutil
from unittest import TestCase
from pipeline.step3_features import generate_features_for_datasets
import os
import time

class TestProcess_features(TestCase):
    test_output_path = "tmp/features/1"
    expected_output_path = "expected/features/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    def test_generates_features(self):
        start = time.process_time()
        generate_features_for_datasets("data", self.test_output_path)
        duration = time.process_time() - start

        # should have 2 datasets
        expected_datasets =  {'capture0_dlib', 'capture1_dlib', 'capture0_opencv', 'capture1_opencv', 'capture2_dlib', 'capture2_opencv', 'capture0_landmark_avg', 'capture2_landmark_avg', 'capture1_landmark_avg'}
        self.assertEqual(set(os.listdir(self.test_output_path)), expected_datasets)

        # should have expected landmark.csv's
        for dataset in expected_datasets:
            self.assertListEqual(
                self.readfile("{}/{}/features.csv".format(self.test_output_path, dataset)),
                self.readfile("{}/{}/features.csv".format(self.expected_output_path, dataset))
            )

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 130)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

