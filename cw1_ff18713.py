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
    
    return (x_trains, x_tests, y_trains, y_tests)

#def get_folds(arr, k):
#    sarr = arr[]

def polynomial_cve(folds, e, k):
    x1, x2, y1, y2 = folds
    cves = []
    for i in range(k):
        cs = least_squares(chebyshev(x1[i], e), y1[i])
        y_hat = chebyshev(x2[i], e).dot(cs)
        cves.append(calculate_error(y2[i], y_hat).mean())
    return np.mean(cves)

def sinusoidal_cve(folds, k):
    x1, x2, y1, y2 = folds
    cves = []
    for i in range(k):
        cs = least_squares(np.column_stack((np.ones(x1[i].shape), np.sin(x1[i]))), y1[i])
        y_hat = np.column_stack((np.ones(x2[i].shape), np.sin(x2[i]))).dot(cs)
        cves.append(calculate_error(y2[i], y_hat).mean())
    return np.mean(cves)

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
        folds = create_folds(xs, ys, k)

        lin_cve = polynomial_cve(folds, 1, k)
        pol_cve = polynomial_cve(folds, 2, k)
        sin_cve = sinusoidal_cve(folds, k)

        new_xs = np.linspace(xs.min(), xs.max(), 100)
        if sin_cve < min(lin_cve, pol_cve):
            print(f"Line segment is sinusoidal")
            cs = least_squares(np.column_stack((np.ones(xs.shape), np.sin(xs))), ys)
            new_ys = np.column_stack((np.ones(new_xs.shape), np.sin(new_xs))).dot(cs)
            y_hat = np.column_stack((np.ones(xs.shape), np.sin(xs))).dot(cs)
        else:
            order = 1 if lin_cve < pol_cve else 2 
            print(f"Line segment is polynomial of order {order}")
            cs = least_squares(chebyshev(xs, order), ys)
            new_ys = chebyshev(new_xs, order).dot(cs)
            y_hat = chebyshev(xs, order).dot(cs)
        
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
