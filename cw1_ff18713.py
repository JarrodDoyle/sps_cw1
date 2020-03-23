from __future__ import print_function

from utilities import *

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

def create_folds(xs, ys, k):
    # Shuffle xs and ys such that they maintain their pairings
    indices = np.arange(xs.shape[0])
    np.random.shuffle(indices)
    shuffled_xs = xs[indices]
    shuffled_ys = ys[indices]

    # Calculate fold size, will give an error if k is not an appropriate value for the given data set
    fold_size = len(xs) / k

    x_tests = []
    x_trains = []
    y_tests = []
    y_trains = []

    for i in range(k):
        r1 = int(i*fold_size)
        r2 = int((i+1)*fold_size)

        x_tests.append(shuffled_xs[r1:r2])
        x_trains.append(np.concatenate((shuffled_xs[:r1], shuffled_xs[r2:])))

        y_tests.append(shuffled_ys[r1:r2])
        y_trains.append(np.concatenate((shuffled_ys[:r1], shuffled_ys[r2:])))
    
    return x_tests, x_trains, y_tests, y_trains

def main(data):
    all_xs, all_ys = data
    number_of_segments = len(all_xs) // 20
    line_segments = []
    total_error = 0

    for i in range(number_of_segments):
        xs = all_xs[i*20:(i+1)*20]
        ys = all_ys[i*20:(i+1)*20]

        # Setup folds for cross-validation.
        k = 10
        x_tests, x_trains, y_tests, y_trains = create_folds(xs, ys, k)

        lowest_cve = np.inf
        best_order = None
        # Linear + polynomial (cubed)
        for j in [1, 3]:
            # K-fold cross-validation
            cves = []
            for l in range(k):
                coefficients = least_squares(chebyshev(x_trains[l], j), y_trains[l])
                y_hat = chebyshev(x_tests[l], j).dot(coefficients)
                cves.append(calculate_error(y_tests[l], y_hat).mean())
            cve = np.mean(cves)
            if cve < lowest_cve:
                lowest_cve = cve
                best_order = j

        # Sinusoidal (y = asin(wx) + b)
        cves = []
        for l in range(k):
            coefficients = least_squares(np.column_stack((np.ones(x_trains[l].shape), np.sin(x_trains[l]))), y_trains[l])
            y_hat = np.column_stack((np.ones(x_tests[l].shape), np.sin(x_tests[l]))).dot(coefficients)
            cves.append(calculate_error(y_tests[l], y_hat).mean())
        cve = np.mean(cves)
        
        new_xs = np.linspace(xs.min(), xs.max(), 100)
        if cve < lowest_cve:
            print(f"Line segment is sinusoidal")
            cs = least_squares(np.column_stack((np.ones(xs.shape), np.sin(xs))), ys)
            new_ys = np.column_stack((np.ones(new_xs.shape), np.sin(new_xs))).dot(cs)
            y_hat = np.column_stack((np.ones(xs.shape), np.sin(xs))).dot(cs)
        else:
            print(f"Line segment is polynomial of order {best_order}")
            cs = least_squares(chebyshev(xs, best_order), ys)
            new_ys = chebyshev(new_xs, best_order).dot(cs)
            y_hat = chebyshev(xs, best_order).dot(cs)
        
        line_segments.append((new_xs, new_ys))
        total_error += calculate_error(ys, y_hat)
    
    return total_error, line_segments

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("No filepath supplied.")
    else:
        file_path = args[0]
        xs, ys = load_points_from_file(file_path)
        error, line_segments = main((xs, ys))      

        if "--plot" in args:
            produce_figure(xs, ys, line_segments)
    
        print(error)
