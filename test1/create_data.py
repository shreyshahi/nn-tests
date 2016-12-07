from itertools import izip
import numpy as np


def write_to_file(x_data, y_data, fname):
    with open(fname, "w") as f:
        f.write("x,y\n")
        for x, y in izip(x_data, y_data):
            f.write("{},{}\n".format(x, y))


def main():
    train_x1 = np.arange(-2, -1, 0.0002)
    train_x2 = np.arange(1, 2, 0.0002)

    train_x = np.concatenate((train_x1, train_x2))
    train_y = train_x ** 2

    test_x = np.arange(-10, 10, 0.001)
    test_y = test_x ** 2
    write_to_file(test_x, test_y, "data/test.csv")
    write_to_file(train_x, train_y, "data/train.csv")


if __name__ == "__main__":
    main()

