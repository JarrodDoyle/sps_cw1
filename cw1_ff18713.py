from __future__ import print_function

import numpy as np
from skimage import data, io, color, transform, exposure
from scipy import stats
import matplotlib.pyplot as plt

import sys
import utilities as utils

def least_squares(xs, ys):
    """Use least squares method to estimate parameters of a function
    
    Args:
        xs: Nx(p+1) matrix where p is the power of the polynomial function. First column is full of 1's. Subsequent columns are x values raised to a power.
        ys: Nx1 matrix of y values.
    """
    return np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)

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

        # As a linear regression
        xs_m = np.column_stack((np.ones(xs.shape), xs))
        cs = least_squares(xs_m, ys)
        e1 = calculate_error(ys, xs_m.dot(cs))
        l1 = (xs, xs_m.dot(cs))

        # As a polynomial regression
        xs_m = np.column_stack((np.ones(xs.shape), xs, xs**2, xs**3))
        cs = least_squares(xs_m, ys)
        e2 = calculate_error(ys, xs_m.dot(cs))

        new_xs = np.linspace(xs.min(), xs.max(), 100)
        new_xs_m = np.column_stack((np.ones(new_xs.shape), new_xs, new_xs**2, new_xs**3))    
        l2 = (new_xs, new_xs_m.dot(cs))

        if e1 <= e2:
            print("Linear")
            line_segments.append(l1)
            total_error += e1
        else:
            print("Polynomial")
            line_segments.append(l2)
            total_error += e2

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
