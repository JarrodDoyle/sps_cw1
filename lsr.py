from __future__ import print_function

import numpy as np
from skimage import data, io, color, transform, exposure
from scipy import stats
import matplotlib.pyplot as plt

import sys
import utilities as utils
from enum import Enum

class LineType(Enum):
    LINEAR = 1
    POLYNOMIAL = 2
    UNKNOWN = 3


def determine_segment_type(xs, ys):
    """
    Determines whether a given line segment is of type linear, polynomial, or unknown.
    """
    pass

def least_squares():
    """
    Uses least squares method to determine estimates of the parameters  𝑎  and  𝑏.
    """
    pass

def calculate_error():
    """
    Calculates the error in the reconstructed signal.
    """
    pass

def produce_figure(xs, ys):
    """
    Visualises the inputted data along with the calculated regression line
    """
    utils.view_data_segments(xs, ys)

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("No filepath supplied.")
    else:
        file_path = args[0]
        xs, ys = utils.load_points_from_file(file_path)
    
    if "--plot" in args:
        produce_figure(xs, ys)