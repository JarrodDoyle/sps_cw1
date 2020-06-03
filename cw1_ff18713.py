from __future__ import print_function

from utilities import *

def least_squares(xs, ys):
    """Return estimated function parameters using least squares method.
    
    Parameters:
        xs: Nx(p+1) matrix where p is the power of the polynomial function. First column is full of 1's. Subsequent columns are x values raised to a power.
        ys: Nx1 matrix of y values.
    """
    return np.linalg.inv(xs.T.dot(xs)).dot(xs.T).dot(ys)

def calculate_error(y, y_hat):
    """Returns the error value of the reconstructed signal."""
    return np.sum((y_hat - y) ** 2)

def produce_figure(xs, ys, line_segments):
    """Displays a plot of the data points and the calculated regression line."""
    plt.scatter(xs, ys)
    for (xs, ys) in line_segments:
        plt.plot(xs, ys, c="r")
    plt.show()

def chebyshev(xs, order):
    new_xs = np.array(np.ones(xs.shape))
    for i in range(order):
        new_xs = np.column_stack((new_xs, xs**(i+1)))
    return new_xs

def shuffle_lists(xs, ys):
    """Returns shuffled xs an ys such that pairings are maintained."""
    indices = np.arange(xs.shape[0])
    np.random.shuffle(indices)
    return xs[indices], ys[indices]

def create_folds(xs, ys, k):
    """Return k sets of test and train data for xs and ys"""
    r = shuffle_lists(xs, ys)
    s_xs, s_ys = np.array_split(r[0], k), np.array_split(r[1], k)
    
    folds = {"x": [], "y": []}
    for i in range(k):
        folds["x"].append([s_xs[i], np.append(s_xs[:i], s_xs[i+1:])])
        folds["y"].append([s_ys[i], np.append(s_ys[:i], s_ys[i+1:])])
    
    return folds

def pol_cve(x1, x2, y1, y2, e):
    cs = least_squares(chebyshev(x1, e), y1)
    y_hat = chebyshev(x2, e).dot(cs)
    return calculate_error(y2, y_hat).mean()

def sin_cve(x1, x2, y1, y2):
    cs = least_squares(np.column_stack((np.ones(x1.shape), np.sin(x1))), y1)
    y_hat = np.column_stack((np.ones(x2.shape), np.sin(x2))).dot(cs)
    return calculate_error(y2, y_hat).mean()

def perform_k_fold_validation(xs, ys, k):
    folds = create_folds(xs, ys, k)
    cves = {"lin": [], "pol": [], "sin": []}
    for i in range(k):
        (x2, x1), (y2, y1) = folds["x"][i], folds["y"][i]
        cves["lin"].append(pol_cve(x1, x2, y1, y2, 1))
        cves["pol"].append(pol_cve(x1, x2, y1, y2, 2))
        cves["sin"].append(sin_cve(x1, x2, y1, y2))
    
    return np.mean(cves["lin"]), np.mean(cves["pol"]), np.mean(cves["sin"])

def main(data, n):
    split_xs, split_ys = np.array_split(data[0], n), np.array_split(data[1], n)
    line_segments = []
    total_error = 0
    
    for (xs, ys) in zip(split_xs, split_ys):
        lin_cve, pol_cve, sin_cve = perform_k_fold_validation(xs, ys, 10)

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
        error, line_segments = main((xs, ys), len(xs) // 20)      

        if "--plot" in args:
            produce_figure(xs, ys, line_segments)
    
        print(error)
