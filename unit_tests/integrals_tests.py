import unittest
import json
import numpy
import os

from integrals.integrals import elliptical_integral_cubic_all_roots_real
from integrals.integrals import elliptical_integral_cubic_one_real_and_two_complex_roots
from integrals.integrals import elliptical_integral_quartic_all_complex_roots
from integrals.integrals import elliptical_integral_quartic_all_real_roots
from integrals.integrals import elliptical_integral_quartic_two_real_and_two_complex_roots
absolute_path = os.getcwd() + '\\unit_tests\\'
# absolute_path = os.getcwd() + '\\'


def open_file(filename):
    with open(os.path.join(absolute_path, filename)) as test_data:
        data = json.load(test_data)
    return data


class TestEllipticalIntegralsQuartic(unittest.TestCase):
    file = 'test_data_elliptic_integral_quartic_cases.json'

    def test_quartic_all_complex(self):
        data = open_file(self.file)
        case = data['test_elliptical_integral_quartic_all_complex_roots']
        self.assertTrue(numpy.allclose(elliptical_integral_quartic_all_complex_roots(
            case['p_list'], case['a_list'], case['b_list'], case['fgh_1_list'], case['fgh_2_list'], case['ffr'],
            case['y_val'], case['x_val']) / 4.0, case['expected_result_value']))

    def test_quartic_all_real(self):
        data = open_file(self.file)
        case = data['test_elliptical_integral_quartic_all_real_roots']
        self.assertTrue(numpy.allclose(elliptical_integral_quartic_all_real_roots(
            case['p_list'], case['a_list'], case['b_list'], case['ffr'], case['y_val'], case['x_val']) / 2.0,
                                       case['expected_result_value']))

    def test_quartic_two_real_two_complex(self):
        data = open_file(self.file)
        case = data['test_elliptical_integral_quartic_two_real_and_two_complex_roots']
        self.assertTrue(numpy.allclose(elliptical_integral_quartic_two_real_and_two_complex_roots(
            case['p_list'], case['a_list'], case['b_list'], case['kwadrat_list'], case['ffr'], case['y_val'],
            case['x_val']) / 4.0, case['expected_result_value']))


class TestEllipticalIntegralsCubic(unittest.TestCase):
    file = 'test_data_elliptic_integral_cubic_cases.json'

    def test_cubic_all_real(self):
        data = open_file(self.file)
        case = data['test_elliptical_integral_cubic_all_roots_real']
        self.assertTrue(numpy.allclose(elliptical_integral_cubic_all_roots_real(
            case['p_list'], case['a_list'], case['b_list'], case['ffr'], case['y_val'], case['x_val']) / 2.0,
                                       case['expected_result_value']))

    def test_cubic_one_real_two_complex(self):
        data = open_file(self.file)
        case = data['test_integral_cubic_one_real_and_two_complex_roots']
        self.assertTrue(numpy.allclose(elliptical_integral_cubic_one_real_and_two_complex_roots(
            case['p_list'], case['a_list'], case['b_list'], case['kwadrat_list'], case['ffr'], case['y_val'],
            case['x_val']) / 4.0, case['expected_result_value']))


if __name__ == '__main__':
    unittest.main()
