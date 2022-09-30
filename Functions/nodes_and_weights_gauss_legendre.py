import numpy as np


def gauss_legendre_nodes_and_weights(lower_boundary: float, upper_boundary: float, number_of_evaluations: int) -> list:
    accuracy = 3.0E-14
    m = number_of_evaluations + 1
    m = m // 2
    x_max = (upper_boundary + lower_boundary) / 2
    x_min = (upper_boundary - lower_boundary) / 2
    nodes = [0] * (number_of_evaluations + 1)
    weights = [0] * (number_of_evaluations + 1)
    for i in range(1, m + 1):
        z = np.cos(np.pi * (i - 0.25) / (number_of_evaluations + 0.5))
        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, number_of_evaluations + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = number_of_evaluations * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if np.abs(z - z1) <= accuracy:
                break
        nodes[i] = x_max - x_min * z
        nodes[number_of_evaluations + 1 - i] = x_max + x_min * z
        weights[i] = 2.0 * x_min / ((1.0 - z ** 2) * pp ** 2)
        weights[number_of_evaluations + 1 - i] = weights[i]
    # This last part is to remove unnecessary 0 in first position of nodes and weights lists
    final_nodes = []
    final_weights = []
    for i in range(len(nodes)):
        if i == 0:
            continue
        final_nodes.append(nodes[i])
        final_weights.append(weights[i])
    return [final_nodes, final_weights]


if __name__ == '__main__':
    a = -1
    b = 1
    num = 10
    n, w = gauss_legendre_nodes_and_weights(a, b, num)
    print(n, w)
