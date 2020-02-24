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
    return LineType.LINEAR

def least_squares(xs, ys, x_func):
    """
    Uses least squares method to determine estimates of the parameters  𝑎  and  𝑏.
    """
    xs = np.column_stack((np.ones(xs.shape), x_func(xs)))
    (a, b) = np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)
    return (a, b)

def calculate_error(y, y_hat):
    """
    Calculates the error in the reconstructed signal.
    """
    return np.sum((y_hat - y) ** 2)

def produce_figure(xs, ys, line_segments):
    """
    Visualises the inputted data along with the calculated regression line
    """
    plt.scatter(xs, ys)
    for (xs, ys) in line_segments:
        plt.plot(xs, ys, c="r")
    plt.show()

def main(data):
    all_xs, all_ys = data
    number_of_segments = len(all_xs) // 20
    line_segments = []
    total_error = 0

    for i in range(number_of_segments):
        xs = all_xs[i*20:(i+1)*20]
        ys = all_ys[i*20:(i+1)*20]

        segment_type = determine_segment_type(xs, ys)
        if segment_type == LineType.LINEAR:
            a, b = least_squares(xs, ys, lambda x: x)
            line_segments.append((xs, ys))
        elif segment_type == LineType.POLYNOMIAL:
            pass
        elif segment_type == LineType.UNKNOWN:
            pass
        
        total_error += calculate_error(ys, a + b * xs)
    return total_error, line_segments

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("No filepath supplied.")
    else:
        file_path = args[0]
        xs, ys = utils.load_points_from_file(file_path)
        error, line_segments = main((xs, ys))      

        if "--plot" in args:
            produce_figure(xs, ys, line_segments)
    
        print(error)