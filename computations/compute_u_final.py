import math
import numpy as np
import scipy
from integrals import integrals as ints

from setup.init_setup import InitSchema
from setup.variables import VariablesSchema


class ComputeUFinal:
    """
        Class responsible for computing u_final value when we know u_start, mu_start and mu_final
        Input variables will be passed in form of dict()
        Input variables contains (keys only):
        {'spin', 'u_start', 'mu_start', 'mu_final', 'angular_momentum', 'carter_c', 'mu_turning_points'}
        Output will be return in for of a dict() as well
    """
    variables: VariablesSchema

    def __init__(self, variables: VariablesSchema):
        self.variables = variables

    def compute_mu_integral(self, case: int) -> float:
        def compute_case_1() -> float:
            s = np.sign(self.variables.MU_START)
            coefficient_a = s * self.variables.MU_INTEGRAL_SIGN
            coefficient_b = coefficient_a * (-1.0) ** (self.variables.MU_TURNING_POINTS + 1)
            mu_plus = s * np.sqrt(1.0 - self.variables.L ** 2.0 / self.variables.SPIN ** 2.0)
            mu_integral_p1 = np.arccosh(mu_plus / self.variables.MU_START) / np.abs(self.variables.SPIN * mu_plus)
            mu_integral_p2 = np.arccosh(mu_plus / self.variables.MU_FINAL) / np.abs(self.variables.SPIN * mu_plus)
            return coefficient_a * mu_integral_p1 + coefficient_b * mu_integral_p2

        def compute_case_2() -> float:
            mu_plus = np.sqrt(self.variables.CARTER_CONST / (self.variables.CARTER_CONST + self.variables.L ** 2.0))
            if self.variables.MU_START > mu_plus:
                mu_plus = self.variables.MU_START
            coefficient_a = self.variables.MU_INTEGRAL_SIGN * (-1.0) ** self.variables.MU_TURNING_POINTS
            coefficient_b = 2.0 * int((2.0 * self.variables.MU_TURNING_POINTS + 3.0 -
                                       self.variables.MU_INTEGRAL_SIGN) / 4.0) - 1.0
            mu_integral_p1 = (0.5 * np.pi - np.arcsin(self.variables.MU_START / mu_plus)) / \
                             np.sqrt(self.variables.CARTER_CONST + self.variables.L ** 2.0)
            mu_integral_p2 = (0.5 * np.pi + np.arcsin(self.variables.MU_FINAL / mu_plus)) / \
                             np.sqrt(self.variables.CARTER_CONST + self.variables.L ** 2.0)
            mu_integral_p3 = np.pi / np.sqrt(self.variables.CARTER_CONST + self.variables.L ** 2.0)
            return self.variables.MU_INTEGRAL_SIGN * mu_integral_p1 + coefficient_a * mu_integral_p2 + \
                   coefficient_b * mu_integral_p3

        def compute_case_3() -> float:
            return 0.0

        def compute_case_4() -> float:
            return 0.0

        if case == 1:
            return compute_case_1()
        if case == 2:
            return compute_case_2()
        raise Exception('Error occurred in class ComputeUFinal in method compute_mu_integral. '
                        'We checked all possible cases and did not return a proper value. Check file compute_u_final, '
                        'line: 23.')

    def compute(self) -> None:
        # we check for edge cases first
        # edge case 1: carter constant = 0
        if self.variables.CARTER_CONST == 0.0:
            if self.variables.CARTER_CONST ** 2.0 >= self.variables.SPIN ** 2.0 or self.variables.MU_START == 0:
                self.variables['MU_INTEGRAL'] = 0.0
                self.variables['U_FINAL'] = -1.0
                self.variables['CASE'] = 1
                return
            # we are going to compute mu integral case: 1
            self.variables['MU_INTEGRAL'] = self.compute_mu_integral(1)
        # edge case 2: spin = 0
        if self.variables.SPIN == 0.0:
            # we are going to compute mu integral case: 2
            self.variables['MU_INTEGRAL'] = self.compute_mu_integral(2)
        # create a list of coefficients in U(u) equation
        coefficient_list = [
            self.variables.SPIN ** 2.0 - self.variables.L ** 2.0 - self.variables.CARTER_CONST,
            2.0 * ((self.variables.SPIN - self.variables.L) ** 2.0 + self.variables.CARTER_CONST),
            -1.0 * self.variables.SPIN ** 2.0 * self.variables.CARTER_CONST
        ]
        # edge case 3: carter constant = 0, spin = angular momentum and mu_start = 0
        if coefficient_list[1] == coefficient_list[2] == 0.0:
            self.variables['U_FINAL'] = -1.0
            self.variables['CASE'] = 4
            self.variables['U_TURNING_POINTS'] = 0
            return
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
                self.variables['U14'][0] = u1
                self.variables['U14'][1] = u2
                self.variables['U14'][2] = u3
                if self.variables.U_START <= u2 or self.variables.U_START >= u3:
                    pass
                else:
                    # we have unphysical real cubic
                    # we must modify input
                    if self.variables.U_INTEGRAL_SIGN == 1.0:
                        self.variables['U_START'] = u3
                    else:
                        self.variables['U_START'] = u2

                if self.variables.U_START <= u2:
                    self.variables['CASE'] = 1
                    if self.variables.COMPUTE_RELEVANT_VARIABLES and self.variables.U_FINAL != u2:
                        self.variables['IU_U0_T'] = ints.elliptical_integral_cubic_all_roots_real(
                            p=list_of_exponents,
                            a=[-u1, u2, u3],
                            b=[1.0, -1.0, -1.0],
                            ffr=self.variables.RF_IU_U0,
                            y=self.variables.U_START,
                            x=u2
                        )
                    elif self.variables.U_START == u2:
                        self.variables['IU_U0_T'] = 0.0
                    # compute arguments for Jacobi elliptic functions cn and dn
                    jarg = 0.5 * math.sqrt((u3 - u1) * coefficient_list[1]) * \
                           (self.variables.MU_INTEGRAL - self.variables.U_INTEGRAL_SIGN *
                            self.variables.IU_U0_T / math.sqrt(coefficient_list[1]))
                    m = (u3 - u2) / (u3 - u1)
                    sn, cn, dn = scipy.special.ellipj(jarg, m)
                    self.variables['U_FINAL'] = u1 + (u3 - u1) * dn ** 2.0 / cn / cn
                    self.variables['U_TURNING_POINTS'] = int(
                        (-np.sign(
                            1.0,
                            self.variables.U_INTEGRAL_SIGN * (self.variables.MU_INTEGRAL + self.variables.IU_U0_T
                                                              / math.sqrt(coefficient_list[1]))
                        )
                         + 1.0) / 2.0
                    )
                    # TODO uncomment this if statement after U and MU computation modules are ready
                    # if self.variables.COMPUTE_PHI_AND_T_PARTS:
                    #     self.variables['IU_T_U1'] = ints.elliptical_integral_cubic_all_roots_real(
                    #         list_of_exponents, [-u1, u2, u3], [1.0, -1.0, -1.0], self.variables.RF_IU_U1,
                    #         self.variables.U_FINAL, u2
                    #     ) / math.sqrt(coefficient_list[1])
                elif self.variables.U_START >= u3:
                    self.variables['CASE'] = 2
                    if self.variables.COMPUTE_RELEVANT_VARIABLES and self.variables.U_FINAL != u3:
                        self.variables['IU_U0_T'] = ints.elliptical_integral_cubic_all_roots_real(
                            p=list_of_exponents,
                            a=[-u1, -u2, -u3],
                            b=[1.0, 1.0, 1.0],
                            ffr=self.variables.RF_IU_U0,
                            y=u3,
                            x=self.variables.U_START
                        )
                    elif self.variables.U_START != u3:
                        self.variables['IU_U0_T'] = 0.0
                    # compute arguments for Jacobi elliptic functions cn and dn
                    jarg = 0.5 * math.sqrt((u3 - u1) * coefficient_list[1]) * \
                           (self.variables.MU_INTEGRAL + self.variables.U_INTEGRAL_SIGN *
                            self.variables.IU_U0_T / math.sqrt(coefficient_list[1]))
                    m = (u3 - u2) / (u3 - u1)
                    sn, cn, dn = scipy.special.ellipj(jarg, m)
                    self.variables['U_FINAL'] = u1 + (u3 - u1) * dn ** 2.0 / cn / cn
                    self.variables['U_TURNING_POINTS'] = int(
                        (-np.sign(
                            1.0,
                            self.variables.U_INTEGRAL_SIGN * (self.variables.MU_INTEGRAL + self.variables.IU_U0_T
                                                              / math.sqrt(coefficient_list[1]))
                        )
                         + 1.0) / 2.0
                    )
                    # TODO uncomment this if statement after U and MU computation modules are ready
                    # if self.variables.COMPUTE_PHI_AND_T_PARTS:
                    #     self.variables['IU_T_U1'] = ints.elliptical_integral_cubic_all_roots_real(
                    #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
                    #         u3, self.variables.U_FINAL
                    #     ) / math.sqrt(coefficient_list[1])
            elif np.abs(discriminant) < 1.0e-16:
                # here we have case with 3 roots when two of them are equal
                self.variables.U_TURNING_POINTS = 0
                u1 = -2.0 * math.sqrt(coef_1) - coefficient_list[0] / coefficient_list[1] / 3.0
                u2 = -2.0 * math.sqrt(coef_1) * math.cos(2.0 * math.pi / 3.0) - \
                     coefficient_list[0] / coefficient_list[1] / 3.0
                u3 = u2
                self.variables['U14'][0] = u1
                self.variables['U14'][1] = u2
                self.variables['U14'][2] = u3
                self.variables['U_TURNING_POINTS'] = 0
                if self.variables.U_START <= u2:
                    self.variables['CASE'] = 1
                    if self.variables.U_FINAL > u2:
                        self.variables['U_FINAL'] = u2
                        self.variables['U_INTEGRAL'] = self.variables.U_INTEGRAL_SIGN * 1.3
                    else:
                        sarg = math.sqrt((self.variables.U_START - u1) / (u2 - u1))
                        jarg = 0.5 * math.sqrt((u2 - u1) * coefficient_list[1]) * self.variables.U_INTEGRAL_SIGN * \
                               self.variables.MU_INTEGRAL + 0.5 * math.log((1.0 + sarg) / (1.0 - sarg))
                        self.variables['U_FINAL'] = u1 + (u2 - u1) * math.tanh(jarg) ** 2.0
                elif self.variables.U_START >= u2:
                    self.variables['CASE'] = 2
                    if self.variables.U_FINAL < u2:
                        self.variables['U_FINAL'] = u2
                        self.variables['U_INTEGRAL'] = self.variables.U_INTEGRAL_SIGN * 1.3
                    else:
                        sarg = math.sqrt((u2 - u1) / (self.variables.U_START - u1))
                        jarg = -0.5 * math.sqrt((u2 - u1) * coefficient_list[1]) * self.variables.U_INTEGRAL_SIGN * \
                               self.variables.MU_INTEGRAL + 0.5 * math.log((1.0 + sarg) / (1.0 - sarg))
                        self.variables['U_FINAL'] = u1 + (u2 - u1) / math.tanh(jarg) ** 2.0
            else:
                # here we have a case with one real and two complex roots
                self.variables['CASE'] = 3
                self.variables.U_TURNING_POINTS = 0
                coefficient_a = -np.sign(1.0, coef_2) * (np.abs(coef_2) * math.sqrt(discriminant)) ** 1.0 / 3.0
                if coefficient_a != 0.0:
                    coefficient_b = coef_1 / coefficient_a
                else:
                    coefficient_b = 0.0
                u1 = (coefficient_a + coefficient_b) - coefficient_list[0] / coefficient_list[1] / 3.0
                self.variables['U14'][0] = u1
                fgh_coefficients_for_elliptic_integral = [-1.0 / coefficient_list[1] / u1,
                                                          -1.0 / coefficient_list[1] / u1 / u1, 1.0]
                # first we check if valid solution exists
                if self.variables.U_INTEGRAL_SIGN > 0:
                    self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_one_real_and_two_complex_roots(
                        p=list_of_exponents,
                        a=[-u1, 0.0, 0.0, 0.0],
                        b=[1.0, 0.0, 0.0, 0.0],
                        fgh=fgh_coefficients_for_elliptic_integral,
                        ffr=self.variables.RF_IU_U1,
                        y=self.variables.U_START,
                        x=self.variables.U_PLUS
                    ) / math.sqrt(coefficient_list[1])
                else:
                    self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_one_real_and_two_complex_roots(
                        p=list_of_exponents,
                        a=[-u1, 0.0, 0.0, 0.0],
                        b=[1.0, 0.0, 0.0, 0.0],
                        fgh=fgh_coefficients_for_elliptic_integral,
                        ffr=self.variables.RF_IU_U1,
                        y=0.0,
                        x=self.variables.U_START
                    ) / math.sqrt(coefficient_list[1])

                if self.variables.MU_INTEGRAL > self.variables.U_INTEGRAL:
                    self.variables['U_FINAL'] = -1.0
                    self.variables['CASE'] = 0
                if self.variables.COMPUTE_RELEVANT_VARIABLES:
                    # TODO zastanowic sie nad tym U_INTEGRAL w ponizszej linii
                    self.variables.U_INTEGRAL = ints.elliptical_integral_cubic_one_real_and_two_complex_roots(
                        p=list_of_exponents,
                        a=[-u1, 0.0, 0.0, 0.0],
                        b=[1.0, 0.0, 0.0, 0.0],
                        fgh=fgh_coefficients_for_elliptic_integral,
                        ffr=self.variables.RF_IU_U1,
                        y=u1,
                        x=self.variables.U_START
                    ) * self.variables.U_INTEGRAL_SIGN
                # temporary coefficients
                c3 = -1.0
                if self.variables.SPIN != 0.0 or self.variables.L != 0.0:
                    c3 = (self.variables.SPIN + self.variables.L) / (self.variables.SPIN - self.variables.L)
                c2 = math.sqrt(3 * u1 * u1 + u1 * c3)
                c1 = math.sqrt(c2 * coefficient_list[1])
                m1 = 0.5 + (6.0 * u1 + c3) / c2 / 8.0
                jarg = c1 * (self.variables.MU_INTEGRAL + self.variables.U_INTEGRAL)
                sn, cn, dn = scipy.special.ellipj(jarg, m1)
                self.variables['U_FINAL'] = (c2 + u1 - (c2 - u1) * cn) / (1.0 + cn)
                # TODO uncomment this if statement after U and MU computation modules are ready, check for correctness
                # if self.variables.COMPUTE_PHI_AND_T_PARTS:
                #     self.variables['IU_T_U1'] = ints.elliptical_integral_cubic_all_roots_real(
                #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
                #         u3, self.variables.U_FINAL
                #     ) / math.sqrt(coefficient_list[1])


if __name__ == '__main__':
    a = ComputeUFinal()
    print(a)
