from typing import Optional, Dict

import numpy as np


class ComputeUFinal:
    """
        Class responsible for computing u_final value when we know u_start, mu_start and mu_final
        Input variables will be passed in form of dict()
        Input variables contains (keys only):
        {'spin', 'u_start', 'mu_start', 'mu_final', 'angular_momentum', 'carter_c', 'mu_turning_points'}
        Output will be return in for of a dict() as well
    """
    def __init__(self, input_variables: dict):
        self.input_variables = input_variables
        self.temporary_variables = {'mu_integral_case': 0}
        self.output = dict()
        # initialize output with Nones
        self.setup_output_dict()

        # set the coefficients for quadratic equation U(u)
        self.c_coefficient = self.input_variables['spin'] ** 2.0 - self.input_variables['angular_momentum'] ** 2.0 \
                             - self.input_variables['carter_c']
        self.d_coefficient = 2.0 * ((self.input_variables['spin'] - self.input_variables['angular_momentum']) ** 2.0
                                    + self.input_variables['carter_c'])
        self.e_coefficient = -1.0 * self.input_variables['spin'] ** 2.0 * self.input_variables['carter_c']

    def setup_output_dict(self) -> None:
        self.output['u_final'] = None
        self.output['mu_integral'] = None

    def compute(self) -> dict:
        # we check for edge cases first
        # edge case 1: carter constant = 0
        if self.input_variables['carter_c'] == 0.0:
            if self.input_variables['carter_c'] ** 2.0 >= self.input_variables['spin'] ** 2.0 or \
                    self.input_variables['mu_start'] == 0.0:
                self.update_out_case_0()
                return self.output
            self.temporary_variables['mu_integral_case'] = 1

    def compute_mu_integral(self, case: int) -> float:
        def compute_case_1() -> float:
            s = np.sign(self.input_variables['mu_start'])
            coefficient_a = s * self.input_variables['sign_of_mu_integral']
            coefficient_b = coefficient_a * (-1.0) ** (self.input_variables['mu_turning_points'] + 1)
            mu_plus = s * np.sqrt(1.0 - self.input_variables['angular_momentum'] ** 2.0 /
                                  self.input_variables['spin'] ** 2.0)
            mu_integral_p1 = np.arccosh(mu_plus / self.input_variables['mu_start']) / \
                             np.abs(self.input_variables['spin'] * mu_plus)
            mu_integral_p2 = np.arccosh(mu_plus / self.input_variables['mu_final']) / \
                             np.abs(self.input_variables['spin'] * mu_plus)
            return coefficient_a * mu_integral_p1 + coefficient_b * mu_integral_p2

        def compute_case_2() -> float:
            mu_plus = np.sqrt(self.input_variables['carter_c'] / (self.input_variables['carter_c'] +
                                                                  self.input_variables['angular_momentum'] ** 2.0))
            if self.input_variables['mu_start'] > mu_plus:
                mu_plus = self.input_variables['mu_start']
            coefficient_a = self.input_variables['sign_of_mu_integral'] * \
                            (-1.0) ** self.input_variables['mu_turning_points']
            coefficient_b = 2.0 * int((2.0 * self.input_variables['mu_turning_points'] + 3.0 -
                                       self.input_variables['sign_of_mu_integral']) / 4.0) - 1.0
            mu_integral_p1 = (0.5 * np.pi - np.arcsin(self.input_variables['mu_start'] / mu_plus)) / \
                             np.sqrt(self.input_variables['carter_c'] + self.input_variables['angular_momentum'] ** 2.0)
            mu_integral_p2 = (0.5 * np.pi + np.arcsin(self.input_variables['mu_final'] / mu_plus)) / \
                             np.sqrt(self.input_variables['carter_c'] + self.input_variables['angular_momentum'] ** 2.0)
            mu_integral_p3 = np.pi / np.sqrt(self.input_variables['carter_c']
                                             + self.input_variables['angular_momentum'] ** 2.0)
            return self.input_variables['sign_of_mu_integral'] * mu_integral_p1 + \
                   coefficient_a * mu_integral_p2 + coefficient_b * mu_integral_p3

        def compute_case_3() -> float:
            return 0.0

        def compute_case_4() -> float:
            return 0.0

        if case == 1:
            return compute_case_1()
        if case == 2:
            return compute_case_2()
        if case == 3:
            return compute_case_3()
        if case == 4:
            return compute_case_4()
        raise Exception('Error occurred in class ComputeUFinal in method compute_mu_integral. '
                        'We checked all possible cases and did not return a proper value. Check file compute_u_final, '
                        'line: 42.')

    def update_out_case_0(self):
        self.output['mu_integral'] = 0.0
        self.output['u_final'] = -1.0
        self.output['case'] = 0


if __name__ == '__main__':
    a = ComputeUFinal()
    print(a)
