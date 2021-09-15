import shutil
from unittest import TestCase
from pipeline2.step2_features import generate_features_for_datasets
import os
import time
import pandas as pd

from pandas.testing import assert_frame_equal


class TestProcess_features(TestCase):
    test_output_path = "tmp/pipeline2/features/1"
    expected_output_path = "expected/pipeline2/features/1"


    def test_generate_features_for_datasets(self):
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)
        start = time.process_time()
        generate_features_for_datasets("data/pipeline2", self.test_output_path)
        duration = time.process_time() - start

        # should have 3 datasets
        expected_datasets = {'capture0', 'capture1', 'capture2'}
        self.assertTrue(set(os.listdir(self.test_output_path)).issuperset(expected_datasets))

        # should have expected landmark.csv's
        for dataset in expected_datasets:
            output_df = pd.read_csv("{}/{}/features.csv".format(self.test_output_path, dataset))
            expected_df = pd.read_csv("{}/{}/features.csv".format(self.expected_output_path, dataset))
            columns = ['image_path','raw_path','landmark_path','rel_target_x','rel_target_y','timestamp',
                       'frame','gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                       'gaze_angle_x', 'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']

            assert_frame_equal(output_df[columns], expected_df[columns], atol=0.1)
            print(f"Dataset {dataset} correct")

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 300)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

