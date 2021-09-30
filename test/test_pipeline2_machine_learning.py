import shutil
from unittest import TestCase
from pipeline2.step3_machine_learning import generate_predictions_for_datasets

import time
import pandas as pd
from pandas.testing import assert_frame_equal
import os

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

        output_df = pd.read_csv("{}/models_predictions.csv".format(self.test_output_path))
        self.assertTrue(
            len(output_df) > 70
        )

        expected_df = pd.read_csv("{}/models_predictions.csv".format(self.expected_output_path))

        assert_frame_equal(output_df, expected_df, atol=0.9)

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 150)


    def test_drawing_predictions(self):
        generate_predictions_for_datasets("data/pipeline2", self.test_output_path)
        predictions_csv = "{}/models_predictions.csv".format(self.test_output_path)
        from libs.prediction_markers import draw_prediction_markers
        draw_prediction_markers(predictions_csv)

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



    def readfile(self, path):
        with open(path) as f:
            return f.readlines()
