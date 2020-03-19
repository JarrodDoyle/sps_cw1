from __future__ import print_function

import numpy as np
from skimage import data, io, color, transform, exposure
from scipy import stats
import matplotlib.pyplot as plt
from math import inf

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

        lowest_cve = inf
        best_order = None
        for j in range(1, 6):
            coefficients = least_squares(chebyshev(x_train, j), y_train)
            y_hat = chebyshev(x_test, j).dot(coefficients)
            cve = calculate_error(y_test, y_hat).mean()
            if cve < lowest_cve:
                lowest_cve = cve
                best_order = j

        print(f"Line segment is polynomial of order {best_order}")
        new_xs = np.linspace(xs.min(), xs.max(), 100)
        cs = least_squares(chebyshev(xs, best_order), ys)
        
        line_segments.append((new_xs, chebyshev(new_xs, best_order).dot(cs)))
        total_error += calculate_error(ys, chebyshev(xs, best_order).dot(cs))
    
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
