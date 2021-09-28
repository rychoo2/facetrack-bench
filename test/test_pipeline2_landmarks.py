import shutil
from unittest import TestCase
from pipeline2.step1_landmarks import generate_landmarks_for_datasets
import os
import time
import pandas as pd

from pandas.testing import assert_frame_equal


class TestProcess_landmarks(TestCase):
    test_output_path = "tmp/pipeline2/landmarks/1"
    expected_output_path = "expected/pipeline2/landmarks/1"


    def test_generate_landmarks_for_datasets(self):
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)
        start = time.process_time()
        generate_landmarks_for_datasets("data/pipeline2", self.test_output_path)
        duration = time.process_time() - start

        # should have 3 datasets
        self.assertTrue(set(os.listdir(self.test_output_path)).issuperset({'capture0', 'capture1', 'capture2'}))

        # should have expected landmark.csv's
        for dataset in ['capture0', 'capture1', 'capture2']:
            output_df = pd.read_csv("{}/{}/landmarks.csv".format(self.test_output_path, dataset))
            expected_df = pd.read_csv("{}/{}/landmarks.csv".format(self.expected_output_path, dataset))
            columns = ['frame','gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y', "landmark_image"]

            assert_frame_equal(output_df[columns], expected_df[columns], atol=0.02)

        self.assertEqual(set(os.listdir("{}/capture0/images".format(self.test_output_path))),
         {'frame_6.jpg',
          'frame_12.jpg',
          'frame_13.jpg',
          'frame_7.jpg',
          'frame_5.jpg',
          'frame_11.jpg',
          'frame_10.jpg',
          'frame_4.jpg',
          'frame_14.jpg',
          'frame_15.jpg',
          'frame_1.jpg',
          'frame_3.jpg',
          'frame_17.jpg',
          'frame_16.jpg',
          'frame_2.jpg',
          'frame_9.jpg',
          'frame_8.jpg'})
        self.assertEqual(len(os.listdir("{}/capture1/images".format(self.test_output_path))), 33)
        self.assertEqual(len(os.listdir("{}/capture2/images".format(self.test_output_path))), 19)

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 300)


    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

