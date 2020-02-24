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


class Line:
    def __init__(self, xs, ys, a, b):
        self.xs = xs
        self.ys = ys
        self.a = a
        self.b = b

def determine_segment_type(xs, ys):
    """
    Determines whether a given line segment is of type linear, polynomial, or unknown.
    """
    return LineType.LINEAR

def least_squares(xs, ys):
    """
    Uses least squares method to determine estimates of the parameters  𝑎  and  𝑏.
    """
    line_type = determine_segment_type(xs, ys)
    if line_type == LineType.LINEAR:
        xs = np.column_stack((np.ones(xs.shape), xs))
        (a, b) = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)
    elif line_type == LineType.POLYNOMIAL:
        pass
    elif line_type == LineType.UNKNOWN:
        pass
    return (a, b)

def calculate_error(y, y_hat):
    """
    Calculates the error in the reconstructed signal.
    """
    return np.sum((y_hat - y) ** 2)

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
        all_xs, all_ys = utils.load_points_from_file(file_path)

        number_of_segments = len(all_xs) // 20
        line_segments = []
        total_error = 0

        for i in range(number_of_segments):
            xs = all_xs[i*20:(i+1)*20]
            ys = all_ys[i*20:(i+1)*20]

            a, b = least_squares(xs, ys)
            error = calculate_error(ys, a + b * xs)

            line_segments.append(Line(xs, ys, a, b))
            total_error += error
    
    if "--plot" in args:
        produce_figure(xs, ys)
    
    print(total_error)