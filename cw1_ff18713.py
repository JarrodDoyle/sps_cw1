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

def chebyshev(xs, order):
    new_xs = np.array(np.ones(xs.shape))
    for i in range(order):
        new_xs = np.column_stack((new_xs, xs**(i+1)))
    return new_xs

def main(data):
    all_xs, all_ys = data
    number_of_segments = len(all_xs) // 20
    line_segments = []
    total_error = 0

    for i in range(number_of_segments):
        xs = all_xs[i*20:(i+1)*20]
        ys = all_ys[i*20:(i+1)*20]

        x_train = xs[:15]
        x_test = xs[15:]

        y_train = ys[:15]
        y_test = ys[15:]

        # As a linear regression
        l_cs = least_squares(chebyshev(x_train, 1), y_train)
        l_ys = (chebyshev(x_test, 1)).dot(l_cs)
        l_cve = calculate_error(y_test, l_ys).mean()

        # As a polynomial regression
        p_cs = least_squares(chebyshev(x_train, 3), y_train)
        p_ys = chebyshev(x_test, 3).dot(p_cs)
        p_cve = calculate_error(y_test, p_ys).mean()

        if l_cve <= p_cve:
            print("linear")
            cs = least_squares(chebyshev(xs, 1), ys)
            y_hat = chebyshev(xs, 1).dot(cs)
            line_segments.append((xs, y_hat))
            total_error += calculate_error(ys, y_hat)
        else:
            print("polynomial")
            new_xs = np.linspace(xs.min(), xs.max(), 100)
            cs = least_squares(chebyshev(xs, 3), ys)
            new_y_hat = chebyshev(new_xs, 3).dot(cs)
            line_segments.append((new_xs, new_y_hat))
            total_error += calculate_error(ys, chebyshev(xs, 3).dot(cs))

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
