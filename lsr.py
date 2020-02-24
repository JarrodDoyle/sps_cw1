from __future__ import print_function

import numpy as np
from skimage import data, io, color, transform, exposure
from scipy import stats
import matplotlib.pyplot as plt

import sys
import utilities as utils

def calculate_x_function(xs, ys):
    """
    Returns a lambda function representing the function applied to x in the given line segment.
    """
    return lambda x : x

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

        x_func = calculate_x_function(xs, ys)
        a, b = least_squares(xs, ys, x_func)
        new_xs = np.linspace(xs.min(), xs.max(), 100)
        new_ys = a + b * x_func(new_xs)
        line_segments.append((new_xs, new_ys))
        
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