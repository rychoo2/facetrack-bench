import shutil
from unittest import TestCase
from pipeline.step1_landmarks import generate_landmarks_for_datasets, generate_landmark_for_file, save_landmark_image
import os
import time
import cv2
from libs.simple_eye_landmark_detector import SimpleEyeLandmarkDetector

class TestProcess_landmarks(TestCase):
    test_output_path = "tmp/landmarks/1"
    expected_output_path = "expected/landmarks/1"


    def test_generate_landmarks_for_datasets(self):
        shutil.rmtree(self.test_output_path, ignore_errors=True, onerror=None)
        start = time.process_time()
        generate_landmarks_for_datasets("data", self.test_output_path)
        duration = time.process_time() - start

        # should have 2 datasets
        self.assertTrue(set(os.listdir(self.test_output_path)).issuperset({'capture0', 'capture1', 'capture2'}))

        # should have expected landmark.csv's
        for dataset in ['capture0', 'capture1', 'capture2']:
            self.assertListEqual(
                self.readfile("{}/{}/landmarks.csv".format(self.test_output_path, dataset)),
                self.readfile("{}/{}/landmarks.csv".format(self.expected_output_path, dataset))
            )

        # should contain images
        self.assertEqual(len(os.listdir("{}/capture0/images".format(self.test_output_path))), 17)
        self.assertEqual(len(os.listdir("{}/capture1/images".format(self.test_output_path))), 36)

        # should be 'quick'
        print("took {} cpu time".format(duration))
        self.assertLess(duration, 250)


    def  test_process_for_file(self):
        img, face = generate_landmark_for_file('images/face1.jpg')
        print(face)

        save_landmark_image(img, face, 'tmp/face1.jpg')
        self.assertIsNotNone(face)
        self.assertTrue(len(face['landmarks_opencv']) > 0 or len(face['landmarks_dlib']) > 0)

    def test_process_for_file2(self):
        img, face = generate_landmark_for_file('images/face0.jpg')
        print(face)

        save_landmark_image(img, face, 'tmp/face0.jpg')
        self.assertIsNotNone(face)
        self.assertTrue(len(face['landmarks_opencv']) > 0 or len(face['landmarks_dlib']) > 0)

    def test_process_for_file3(self):
        img, face = generate_landmark_for_file('images/face2.jpg')
        print(face)

        save_landmark_image(img, face, 'tmp/face2.jpg')
        self.assertIsNotNone(face)
        self.assertTrue(len(face['landmarks_opencv']) > 0 or len(face['landmarks_dlib']) > 0)
        self.assertTrue('pupil' in face['right_eye'])
        self.assertTrue('pupil' in face['left_eye'])

    def test_process_for_file3_mtcnn(self):
        img, face = generate_landmark_for_file('images/face2.jpg')
        print(face)

        save_landmark_image(img, face, 'tmp/face2.jpg')
        self.assertIsNotNone(face)
        self.assertTrue(len(face['landmarks_mtcnn']) > 0)

    def test_process_for_file3(self):
        img, face = generate_landmark_for_file('images/face3.jpg')
        print(face)

        save_landmark_image(img, face, 'tmp/face3.jpg')
        self.assertIsNotNone(face)
        self.assertTrue(len(face['landmarks_opencv']) > 0 and len(face['landmarks_dlib']) > 0)

    def test_process_for_file4(self):
        img, face = generate_landmark_for_file('images/face4.jpg')
        print(face)

        save_landmark_image(img, face, 'tmp/face4.jpg')
        self.assertIsNotNone(face)
        self.assertTrue(len(face['landmarks_opencv']) > 0 or len(face['landmarks_dlib']) > 0)

    def test_eye_landmark_file1(self):

        pupil = SimpleEyeLandmarkDetector().get_landmarks(cv2.imread('images/test_left_eye.jpg'))
        print(pupil)

        self.assertIsNotNone(pupil)

    def test_eye_landmark_file2(self):
        pupil = SimpleEyeLandmarkDetector().get_landmarks(cv2.imread('images/test_right_eye.png'))
        print(pupil)

        self.assertIsNotNone(pupil)

    def readfile(self, path):
        with open(path) as f:
            return f.readlines()

