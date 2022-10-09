from typing import Any
import numpy as np


class ComputeUFinal:
    """
        Class responsible for computing u_final value when we know u_start, mu_start and mu_final
        Input variables will be passed in form of dict()
        Input variables contains (keys only):
        {'spin', 'u_start', 'mu_start', 'mu_final', 'angular_momentum', 'carter_c', 'mu_turning_points'}
        Output will be return in for of a dict() as well
    """
    def __init__(self, input_variables: dict[str, Any]):
        self.input_variables = input_variables
        self.output = dict()
        # initialize output with Nones
        self.setup_output_dict()

    def setup_output_dict(self) -> None:
        self.output['u_final'] = None
        self.output['mu_integral'] = None
        self.output['case'] = None
        self.output['u_turning_points'] = None

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

    def update_output(self, case: int) -> None:
        if case == 1:
            self.output['mu_integral'] = 0.0
            self.output['u_final'] = -1.0
            self.output['case'] = 0
        if case == 2:
            self.output['u_final'] = -1.0
            self.output['case'] = 4
            self.output['u_turning_points'] = 0

    def compute(self) -> dict[str, Any]:
        # we check for edge cases first
        # edge case 1: carter constant = 0
        if self.input_variables['carter_c'] == 0.0:
            if self.input_variables['carter_c'] ** 2.0 >= self.input_variables['spin'] ** 2.0 or \
                    self.input_variables['mu_start'] == 0.0:
                self.update_output(1)
                return self.output
            # we are going to compute mu integral case: 1
            self.output['mu_integral'] = self.compute_mu_integral(1)
        # edge case 2: spin = 0
        if self.input_variables['spin'] == 0.0:
            # we are going to compute mu integral case: 2
            self.output['mu_integral'] = self.compute_mu_integral(2)
        # create a list of coefficients in U(u) equation
        # c_coefficient = coefficient_list[0]
        # d_coefficient = coefficient_list[1]
        # e_coefficient = coefficient_list[2]
        coefficient_list = [
            self.input_variables['spin'] ** 2.0 - self.input_variables['angular_momentum'] ** 2.0
            - self.input_variables['carter_c'],
            2.0 * ((self.input_variables['spin'] - self.input_variables['angular_momentum']) ** 2.0
                   + self.input_variables['carter_c']),
            -1.0 * self.input_variables['spin'] ** 2.0 * self.input_variables['carter_c']
        ]
        # edge case 3: carter constant = 0, spin = angular momentum and mu_start = 0
        if coefficient_list[1] == coefficient_list[2] == 0.0:
            self.update_output(2)
            return self.output
        # after basic edge cases we will solve equation U(u)
        # we have two cases here:
        # 1) U(u) is a third order polynomial
        # 2) U(u) is a fourth order polynomial

        # U(u) is a third order polynomial


if __name__ == '__main__':
    a = ComputeUFinal()
    print(a)
