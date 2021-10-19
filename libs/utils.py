import datetime
import os
import getpass
import numpy as np
import math
from dotenv import load_dotenv, find_dotenv
load_dotenv(dotenv_path=find_dotenv(filename=os.path.join(os.path.dirname(__file__),'../config/{}.env'.format(getpass.getuser()))))

def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]


def get_latest_features(input_root):
    return get_latest(input_root, 'features')


def get_latest_landmarks(input_root):
    return get_latest(input_root, 'landmarks')


def get_latest(input_root, type):
    landmark_root_path = "{}/{}".format(input_root, type)
    last_generation = sorted(os.listdir(landmark_root_path), reverse=True)[0]
    landmark_input_path = "{}/{}".format(landmark_root_path, last_generation)
    return landmark_input_path, list_dirs(landmark_input_path)

def get_datasets(input_root):
    raw_input_path = "{}/raw".format(input_root)
    return raw_input_path, list_dirs(raw_input_path)

def list_dirs(path):
    return [x for x in os.listdir(path) if not '.DS_Store' in x]


# A Halton sequence is a sequence of points within a unit dim-cube which
# have low discrepancy (that is, they appear to be randomish but cover
# the domain uniformly)
def halton(dim: int, nbpts: int):
    h = np.full(nbpts * dim, np.nan)
    p = np.full(nbpts, np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)


# Generate list of n points xy coordinates for given screen dimensions
def points(n, screen_w, screen_h):
    hlt = halton(2, n)
    points = (hlt*(screen_w, screen_h)).astype(int)
    return points.tolist()