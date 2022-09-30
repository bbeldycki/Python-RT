import unittest
import json
import numpy
import os

from Functions.polynomial_roots import find_roots_of_polynomial
from Functions.nodes_and_weights_gauss_legendre import gauss_legendre_nodes_and_weights
absolute_path = os.getcwd() + '\\unit_tests\\'
# absolute_path = os.getcwd() + '\\'


def open_file(filename):
    with open(os.path.join(absolute_path, filename)) as test_data:
        data = json.load(test_data)
    return data


class TestFunctionPolynomialRoots(unittest.TestCase):

    def test_find_roots_of_polynomial(self):
        data = open_file('test_data_roots.json')
        for value in data.values():
            self.assertTrue(numpy.allclose(find_roots_of_polynomial(value['coefficients']),
                                           [value['expected_roots_real_path'],
                                            value['expected_roots_imaginary_path']]))


class TestFunctionMuRoots(unittest.TestCase):
    def test_find_roots_of_polynomial(self):
        data = open_file('test_data_roots.json')
        for value in data.values():
            self.assertTrue(numpy.allclose(find_roots_of_polynomial(value['coefficients']),
                                           [value['expected_roots_real_path'],
                                            value['expected_roots_imaginary_path']]))


class TestFunctionNodesAndWeightsGaussLegendre(unittest.TestCase):
    def test_find_nodes_and_weights_gauss_legendre(self):
        data = open_file('test_data_gauss_legendre_nodes_weights.json')
        for value in data.values():
            self.assertTrue(numpy.allclose(gauss_legendre_nodes_and_weights(value['left_range_of_interval'],
                                           value['right_range_of_interval'],
                                           value['polynomials_degree']),
                            [value['expected_nodes'], value['expected_weights']]))


if __name__ == '__main__':
    unittest.main()
