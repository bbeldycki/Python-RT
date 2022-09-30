import numpy as np


def find_roots_of_polynomial(coefficient: list) -> list:
    solution = np.roots(coefficient)
    return [solution.real.tolist(), solution.imag.tolist()]


if __name__ == '__main__':
    coefficients = [1, 0, -5, 0, 4]
    real_part, img_part = find_roots_of_polynomial(coefficients)
