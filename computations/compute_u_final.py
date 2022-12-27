import math
from typing import Any
import numpy as np

from setup.init_setup import Schema


class ComputeUFinal:
    """
        Class responsible for computing u_final value when we know u_start, mu_start and mu_final
        Input variables will be passed in form of dict()
        Input variables contains (keys only):
        {'spin', 'u_start', 'mu_start', 'mu_final', 'angular_momentum', 'carter_c', 'mu_turning_points'}
        Output will be return in for of a dict() as well
    """
    input_variables: Schema

    def __init__(self, input_data: Schema):
        self.input_variables = input_data
        self.output = {}
        # initialize output with Nones
        self.setup_output_dict()

    def setup_output_dict(self) -> None:
        self.output['u_final'] = None
        self.output['mu_integral'] = None
        self.output['case'] = None
        self.output['u_turning_points'] = None
        self.output['mu_turning_points'] = None

    def compute_mu_integral(self, case: int) -> float:
        def compute_case_1() -> float:
            s = np.sign(self.input_variables.MU_START)
            coefficient_a = s * self.input_variables.MU_INTEGRAL_SIGN
            coefficient_b = coefficient_a * (-1.0) ** (self.input_variables.MU_TURNING_POINTS + 1)
            mu_plus = s * np.sqrt(1.0 - self.input_variables.L ** 2.0 / self.input_variables.SPIN ** 2.0)
            mu_integral_p1 = np.arccosh(mu_plus / self.input_variables.MU_START) / \
                             np.abs(self.input_variables.SPIN * mu_plus)
            mu_integral_p2 = np.arccosh(mu_plus / self.input_variables.MU_FINAL) / \
                             np.abs(self.input_variables.SPIN * mu_plus)
            return coefficient_a * mu_integral_p1 + coefficient_b * mu_integral_p2

        def compute_case_2() -> float:
            mu_plus = np.sqrt(self.input_variables.CARTER_CONST / (self.input_variables.CARTER_CONST +
                                                                   self.input_variables.L ** 2.0))
            if self.input_variables.MU_START > mu_plus:
                mu_plus = self.input_variables.MU_START
            coefficient_a = self.input_variables.MU_INTEGRAL_SIGN * (-1.0) ** self.input_variables.MU_TURNING_POINTS
            coefficient_b = 2.0 * int((2.0 * self.input_variables.MU_TURNING_POINTS + 3.0 -
                                       self.input_variables.MU_INTEGRAL_SIGN) / 4.0) - 1.0
            mu_integral_p1 = (0.5 * np.pi - np.arcsin(self.input_variables.MU_START / mu_plus)) / \
                             np.sqrt(self.input_variables.CARTER_CONST + self.input_variables.L ** 2.0)
            mu_integral_p2 = (0.5 * np.pi + np.arcsin(self.input_variables.MU_FINAL / mu_plus)) / \
                             np.sqrt(self.input_variables.CARTER_CONST + self.input_variables.L ** 2.0)
            mu_integral_p3 = np.pi / np.sqrt(self.input_variables.CARTER_CONST + self.input_variables.L ** 2.0)
            return self.input_variables.MU_INTEGRAL_SIGN * mu_integral_p1 + coefficient_a * mu_integral_p2 + \
                   coefficient_b * mu_integral_p3

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
            self.output['case'] = 1
        if case == 4:
            self.output['u_final'] = -1.0
            self.output['case'] = 4
            self.output['u_turning_points'] = 0

    def compute(self) -> dict[str, Any]:
        # we check for edge cases first
        # edge case 1: carter constant = 0
        if self.input_variables.CARTER_CONST == 0.0:
            if self.input_variables.CARTER_CONST ** 2.0 >= self.input_variables.SPIN ** 2.0 or \
                    self.input_variables.MU_START == 0:
                self.update_output(1)
                return self.output
            # we are going to compute mu integral case: 1
            self.output['mu_integral'] = self.compute_mu_integral(1)
        # edge case 2: spin = 0
        if self.input_variables.SPIN == 0.0:
            # we are going to compute mu integral case: 2
            self.output['mu_integral'] = self.compute_mu_integral(2)
        # create a list of coefficients in U(u) equation
        # c_coefficient = coefficient_list[0]
        # d_coefficient = coefficient_list[1]
        # e_coefficient = coefficient_list[2]
        coefficient_list = [
            self.input_variables.SPIN ** 2.0 - self.input_variables.L ** 2.0 - self.input_variables.CARTER_CONST,
            2.0 * ((self.input_variables.SPIN - self.input_variables.L) ** 2.0 + self.input_variables.CARTER_CONST),
            -1.0 * self.input_variables.SPIN ** 2.0 * self.input_variables.CARTER_CONST
        ]
        # edge case 3: carter constant = 0, spin = angular momentum and mu_start = 0
        if coefficient_list[1] == coefficient_list[2] == 0.0:
            self.update_output(4)
            return self.output
        # after basic edge cases we will solve equation U(u)
        # we have two cases here:
        # 1) U(u) is a third order polynomial
        # 2) U(u) is a fourth order polynomial
        # we start with U(u) as a third order polynomial
        # we are searching for roots of U(u) <=> solutions of equation U(u) = 0
        if coefficient_list[2] == 0.0 and coefficient_list[1] != 0:
            list_of_exponents = [-1, -1, -1, 0, 0]
            # tree special coefficients for needed for solutions
            coef_1 = coefficient_list[0] ** 2.0 / coefficient_list[1] ** 2.0 / 9.0
            coef_2 = (2.0 * coefficient_list[0] ** 3.0 / coefficient_list[1] ** 3.0 + 27.0 / coefficient_list[1]) / 54.0
            discriminant = coef_2 ** 2.0 - coef_1 ** 3.0
            if discriminant < -1.0e-16:
                # we have a case when there are 3 real roots
                # moreover the roots satisfy the relation u1 < 0 < u2 <= u3
                angle_theta = math.acos(coef_2 / coef_1 ** 1.5)
                u1 = -2.0 * math.sqrt(coef_1) * math.cos(angle_theta / 3.0) - \
                     coefficient_list[0] / coefficient_list[1] / 3.0
                u2 = -2.0 * math.sqrt(coef_1) * math.cos((angle_theta - 2.0 * math.pi) / 3.0) - \
                     coefficient_list[0] / coefficient_list[1] / 3.0
                u3 = -2.0 * math.sqrt(coef_1) * math.cos((angle_theta + 2.0 * math.pi) / 3.0) - \
                     coefficient_list[0] / coefficient_list[1] / 3.0
                if self.input_variables.U_START <= u2 or self.input_variables.U_START >= u3:
                    pass
                else:
                    # we have unphysical real cubic
                    # we must modify input
                    if self.input_variables.U_INTEGRAL_SIGN == 1.0:
                        self.input_variables['U_START'] = u3
                    else:
                        self.input_variables['U_START'] = u2

                if self.input_variables.U_START <= u2:
                    pass
                elif self.input_variables.U_START >= u3:
                    pass
            elif abs(discriminant) < 1.0e-16:
                # here we have case with 3 roots when two of them are equal
                self.input_variables.U_TURNING_POINTS = 0
                u1 = -2.0 * math.sqrt(coef_1) - coefficient_list[0] / coefficient_list[1] / 3.0
                u2 = -2.0 * math.sqrt(coef_1) * math.cos(2.0 * math.pi / 3.0) - \
                     coefficient_list[0] / coefficient_list[1] / 3.0
                u3 = u2
                if self.input_variables.U_START <= u2:
                    case = 1

                elif self.input_variables.U_START >= u2:
                    case = 2
            else:
                # here we have a case with one real and two complex roots
                case = 3
                self.input_variables.U_TURNING_POINTS = 0


if __name__ == '__main__':
    a = ComputeUFinal({})
    print(a)
