import shutil
from unittest import TestCase
from pipeline.step1_landmarks import generate_landmarks_for_datasets, generate_landmark_for_file, generate_landmark_image
import os
import time

class TestProcess_landmarks(TestCase):
    test_output_path = "tmp/landmarks/1"
    expected_output_path = "expected/landmarks/1"

    def setUp(self) -> None:
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)

    def test_generate_landmarks_for_datasets(self):
        start = time.process_time()
        generate_landmarks_for_datasets("data", self.test_output_path)
        duration = time.process_time() - start

        # should have 2 datasets
        self.assertTrue(set(os.listdir(self.test_output_path)).issuperset({'capture0', 'capture1'}))

        # should have expected landmark.csv's
        for dataset in ['capture0', 'capture1']:
            self.assertListEqual(
                self.readfile("{}/{}/landmarks.csv".format(self.test_output_path, dataset)),
                self.readfile("{}/{}/landmarks.csv".format(self.expected_output_path, dataset))
            )

        # should contain images
        self.assertEqual(len(os.listdir("{}/capture0/images".format(self.test_output_path))), 17)
        self.assertEqual(len(os.listdir("{}/capture1/images".format(self.test_output_path))), 36)

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 170)


    def test_process_landmarks_for_file(self):
        img, landmarks = generate_landmark_for_file('images/20191215123728.jpg')
        generate_landmark_image(img, landmarks, 'tmp/20191215123728.jpg')

        self.assertIsNotNone(landmarks)
        self.assertTrue(len(landmarks) > 0)

    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

