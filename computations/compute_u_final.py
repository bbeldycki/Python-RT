import math
import numpy as np
import scipy
from integrals import integrals as ints
from integrals import mu_integrals as muints
import Functions.polynomial_roots as polyroots

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
        self.coefficient_list = [
            self.variables.SPIN ** 2.0 - self.variables.L ** 2.0 - self.variables.CARTER_CONST,
            2.0 * ((self.variables.SPIN - self.variables.L) ** 2.0 + self.variables.CARTER_CONST),
            -1.0 * self.variables.SPIN ** 2.0 * self.variables.CARTER_CONST
        ]

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

        def compute_roots_of_mu() -> tuple[float, float, float]:
            solution = (self.variables.SPIN * self.variables.SPIN - self.variables.L * self.variables.L -
                        self.variables.CARTER_CONST * self.variables.CARTER_CONST +
                        np.sign(1.0, self.variables.SPIN * self.variables.SPIN - self.variables.L * self.variables.L -
                                self.variables.CARTER_CONST * self.variables.CARTER_CONST) *
                        math.sqrt((self.variables.SPIN * self.variables.SPIN - self.variables.L * self.variables.L -
                                   self.variables.CARTER_CONST * self.variables.CARTER_CONST) ** 2.0 + 4.0 *
                                  self.variables.SPIN * self.variables.SPIN * self.variables.CARTER_CONST *
                                  self.variables.CARTER_CONST)
                        ) * 0.5 * (-1)
            if (self.variables.SPIN * self.variables.SPIN - self.variables.L * self.variables.L -
                self.variables.CARTER_CONST * self.variables.CARTER_CONST) < 0.0:
                mu_negative = -1 * solution / self.variables.SPIN / self.variables.SPIN
                mu_positive = self.variables.CARTER_CONST * self.variables.CARTER_CONST / solution
            else:
                mu_negative = self.variables.CARTER_CONST * self.variables.CARTER_CONST / solution
                mu_positive = -1 * solution / self.variables.SPIN / self.variables.SPIN
            return mu_negative, mu_positive, math.sqrt(mu_positive)

        def compute_case_3() -> float:
            mu_minus, mu_pos, mu_plus = compute_roots_of_mu()
            a1 = self.variables.MU_INTEGRAL_SIGN
            a2 = self.variables.MU_INTEGRAL_SIGN * (-1.0) ** (self.variables.MU_TURNING_POINTS + 1)
            a3 = 2.0 * int((2.0 * self.variables.MU_TURNING_POINTS - self.variables.MU_INTEGRAL_SIGN + 1) / 4.0)
            if mu_minus < 0.0:
                # symmetric root case
                # orbit can cross equatorial plane
                if self.variables.MU_START > mu_plus:
                    mu_plus = self.variables.MU_START
                    mu_pos = mu_plus * mu_plus
                if self.variables.COMPUTE_RELEVANT_VARIABLES:
                    i1mu, i3mu, self.variables['RF_IMU_MU1'], self.variables['RF_IMU_MU3'] = \
                        muints.mu_integral_symmetric_case_involving_mu_start(
                            spin=self.variables.SPIN,
                            mu_negative=mu_minus,
                            mu_positive=mu_pos,
                            mu_plus=mu_plus,
                            initial_mu=self.variables.MU_START
                        )
                else:
                    # I assign the value for those two variables to make sure that further computation will pass
                    # without any issues
                    i1mu = 0.0
                    i3mu = 0.0
                self.variables['RF_IMU_MU2'], i2mu = muints.mu_integral_symmetric_case_involving_mu_final(
                    spin=self.variables.SPIN,
                    mu_negative=mu_minus,
                    mu_positive=mu_pos,
                    mu_plus=mu_plus,
                    final_mu=self.variables.MU_FINAL,
                    imum=i3mu
                )
            else:
                if np.abs(self.variables.MU_FINAL) < math.sqrt(mu_minus):
                    return -1.0
                else:
                    # TODO przy testowaniu sprawdzic czy np.sign zwraca float czy int
                    if np.sign(1.0, self.variables.MU_START) == -1.0:
                        mu_plus = -1 * mu_plus
                    if np.abs(mu_plus) < np.abs(self.variables.MU_START):
                        mu_plus = self.variables.MU_START
                        mu_pos = mu_plus * mu_plus
                    mu_minus = min(self.variables.MU_START * self.variables.MU_START, mu_minus)
                    if self.variables.COMPUTE_RELEVANT_VARIABLES:
                        i1mu, i3mu, self.variables['RF_IMU_MU1'], self.variables['RF_IMU_MU3'] = \
                            muints.mu_integral_asymmetric_case_involving_mu_start(
                                spin=self.variables.SPIN,
                                mu_negative=mu_minus,
                                mu_positive=mu_pos,
                                mu_plus=mu_plus,
                                initial_mu=self.variables.MU_START
                            )
                    else:
                        # I assign the value for those two variables to make sure that further computation will pass
                        # without any issues
                        i1mu = 0.0
                        i3mu = 0.0
                    self.variables['RF_IMU_MU2'], i2mu = muints.mu_integral_asymmetric_case_involving_mu_final(
                        spin=self.variables.SPIN,
                        mu_negative=mu_minus,
                        mu_positive=mu_pos,
                        mu_plus=mu_plus,
                        final_mu=self.variables.MU_FINAL
                    )
            return a1 * i1mu + a2 * i2mu + a3 * i3mu

        if case == 1:
            return compute_case_1()
        if case == 2:
            return compute_case_2()
        if case == 3:
            return compute_case_3()
        raise Exception('Error occurred in class ComputeUFinal in method compute_mu_integral. '
                        'We checked all possible cases and did not return a proper value. Check file compute_u_final, '
                        f'line: 24, case {case}.')

    def edge_cases(self) -> None:
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
        # edge case 3: carter constant = 0, spin = angular momentum and mu_start = 0
        if self.coefficient_list[1] == self.coefficient_list[2] == 0.0:
            self.variables['U_FINAL'] = -1.0
            self.variables['CASE'] = 4
            self.variables['U_TURNING_POINTS'] = 0
            return

    def cubic_cases(self) -> None:
        list_of_exponents = [-1, -1, -1, 0, 0]
        # tree special coefficients needed for solutions
        coef_1 = self.coefficient_list[0] ** 2.0 / self.coefficient_list[1] ** 2.0 / 9.0
        coef_2 = (2.0 * self.coefficient_list[0] ** 3.0 / self.coefficient_list[1] ** 3.0 + 27.0 /
                  self.coefficient_list[1]) / 54.0
        discriminant = coef_2 ** 2.0 - coef_1 ** 3.0
        if discriminant < -1.0e-16:
            # we have a case when there are 3 real roots
            # moreover the roots satisfy the relation u1 < 0 < u2 <= u3
            angle_theta = math.acos(coef_2 / coef_1 ** 1.5)
            u1 = -2.0 * math.sqrt(coef_1) * math.cos(angle_theta / 3.0) - \
                 self.coefficient_list[0] / self.coefficient_list[1] / 3.0
            u2 = -2.0 * math.sqrt(coef_1) * math.cos((angle_theta - 2.0 * math.pi) / 3.0) - \
                 self.coefficient_list[0] / self.coefficient_list[1] / 3.0
            u3 = -2.0 * math.sqrt(coef_1) * math.cos((angle_theta + 2.0 * math.pi) / 3.0) - \
                 self.coefficient_list[0] / self.coefficient_list[1] / 3.0
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
                jarg = 0.5 * math.sqrt((u3 - u1) * self.coefficient_list[1]) * \
                       (self.variables.MU_INTEGRAL - self.variables.U_INTEGRAL_SIGN *
                        self.variables.IU_U0_T / math.sqrt(self.coefficient_list[1]))
                m = (u3 - u2) / (u3 - u1)
                sn, cn, dn = scipy.special.ellipj(jarg, m)
                self.variables['U_FINAL'] = u1 + (u3 - u1) * dn ** 2.0 / cn / cn
                self.variables['U_TURNING_POINTS'] = int(
                    (-np.sign(
                        1.0,
                        self.variables.U_INTEGRAL_SIGN * (self.variables.MU_INTEGRAL + self.variables.IU_U0_T
                                                          / math.sqrt(self.coefficient_list[1]))
                    )
                     + 1.0) / 2.0
                )
                # TODO uncomment this if statement after U and MU computation modules are ready
                # if self.variables.COMPUTE_PHI_AND_T_PARTS:
                #     self.variables['IU_T_U1'] = ints.elliptical_integral_cubic_all_roots_real(
                #         list_of_exponents, [-u1, u2, u3], [1.0, -1.0, -1.0], self.variables.RF_IU_U1,
                #         self.variables.U_FINAL, u2
                #     ) / math.sqrt(self.coefficient_list[1])
                return
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
                jarg = 0.5 * math.sqrt((u3 - u1) * self.coefficient_list[1]) * \
                       (self.variables.MU_INTEGRAL + self.variables.U_INTEGRAL_SIGN *
                        self.variables.IU_U0_T / math.sqrt(self.coefficient_list[1]))
                m = (u3 - u2) / (u3 - u1)
                sn, cn, dn = scipy.special.ellipj(jarg, m)
                self.variables['U_FINAL'] = u1 + (u3 - u1) * dn ** 2.0 / cn / cn
                self.variables['U_TURNING_POINTS'] = int(
                    (-np.sign(
                        1.0,
                        self.variables.U_INTEGRAL_SIGN * (self.variables.MU_INTEGRAL + self.variables.IU_U0_T
                                                          / math.sqrt(self.coefficient_list[1]))
                    )
                     + 1.0) / 2.0
                )
                # TODO uncomment this if statement after U and MU computation modules are ready
                # if self.variables.COMPUTE_PHI_AND_T_PARTS:
                #     self.variables['IU_T_U1'] = ints.elliptical_integral_cubic_all_roots_real(
                #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
                #         u3, self.variables.U_FINAL
                #     ) / math.sqrt(self.coefficient_list[1])
                return
        elif np.abs(discriminant) < 1.0e-16:
            # here we have case with 3 roots when two of them are equal
            self.variables.U_TURNING_POINTS = 0
            u1 = -2.0 * math.sqrt(coef_1) - self.coefficient_list[0] / self.coefficient_list[1] / 3.0
            u2 = -2.0 * math.sqrt(coef_1) * math.cos(2.0 * math.pi / 3.0) - \
                 self.coefficient_list[0] / self.coefficient_list[1] / 3.0
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
                    return
                else:
                    sarg = math.sqrt((self.variables.U_START - u1) / (u2 - u1))
                    jarg = 0.5 * math.sqrt((u2 - u1) * self.coefficient_list[1]) * self.variables.U_INTEGRAL_SIGN \
                           * self.variables.MU_INTEGRAL + 0.5 * math.log((1.0 + sarg) / (1.0 - sarg))
                    self.variables['U_FINAL'] = u1 + (u2 - u1) * math.tanh(jarg) ** 2.0
                    return
            elif self.variables.U_START >= u2:
                self.variables['CASE'] = 2
                if self.variables.U_FINAL < u2:
                    self.variables['U_FINAL'] = u2
                    self.variables['U_INTEGRAL'] = self.variables.U_INTEGRAL_SIGN * 1.3
                    return
                else:
                    sarg = math.sqrt((u2 - u1) / (self.variables.U_START - u1))
                    jarg = -0.5 * math.sqrt((u2 - u1) * self.coefficient_list[1]) * self.variables.U_INTEGRAL_SIGN \
                           * self.variables.MU_INTEGRAL + 0.5 * math.log((1.0 + sarg) / (1.0 - sarg))
                    self.variables['U_FINAL'] = u1 + (u2 - u1) / math.tanh(jarg) ** 2.0
                    return
        else:
            # here we have a case with one real and two complex roots
            self.variables['CASE'] = 3
            self.variables.U_TURNING_POINTS = 0
            coefficient_a = -np.sign(1.0, coef_2) * (np.abs(coef_2) * math.sqrt(discriminant)) ** 1.0 / 3.0
            if coefficient_a != 0.0:
                coefficient_b = coef_1 / coefficient_a
            else:
                coefficient_b = 0.0
            u1 = (coefficient_a + coefficient_b) - self.coefficient_list[0] / self.coefficient_list[1] / 3.0
            self.variables['U14'][0] = u1
            fgh_coefficients_for_elliptic_integral = [-1.0 / self.coefficient_list[1] / u1,
                                                      -1.0 / self.coefficient_list[1] / u1 / u1, 1.0]
            # first we check if valid solution exists
            if self.variables.U_INTEGRAL_SIGN > 0:
                self.variables['IU_T'] = ints.elliptical_integral_cubic_one_real_and_two_complex_roots(
                    p=list_of_exponents,
                    a=[-u1, 0.0, 0.0, 0.0],
                    b=[1.0, 0.0, 0.0, 0.0],
                    fgh=fgh_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=self.variables.U_START,
                    x=self.variables.U_PLUS
                ) / math.sqrt(self.coefficient_list[1])
            else:
                self.variables['IU_T'] = ints.elliptical_integral_cubic_one_real_and_two_complex_roots(
                    p=list_of_exponents,
                    a=[-u1, 0.0, 0.0, 0.0],
                    b=[1.0, 0.0, 0.0, 0.0],
                    fgh=fgh_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=0.0,
                    x=self.variables.U_START
                ) / math.sqrt(self.coefficient_list[1])

            if self.variables.MU_INTEGRAL > self.variables.U_INTEGRAL:
                self.variables['U_FINAL'] = -1.0
                self.variables['CASE'] = 0
            if self.variables.COMPUTE_RELEVANT_VARIABLES:
                # TODO zastanowic sie nad tym U_INTEGRAL w ponizszej linii
                self.variables['IU0'] = ints.elliptical_integral_cubic_one_real_and_two_complex_roots(
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
            c1 = math.sqrt(c2 * self.coefficient_list[1])
            m1 = 0.5 + (6.0 * u1 + c3) / c2 / 8.0
            jarg = c1 * (self.variables.MU_INTEGRAL + self.variables.U_INTEGRAL)
            sn, cn, dn = scipy.special.ellipj(jarg, m1)
            self.variables['U_FINAL'] = (c2 + u1 - (c2 - u1) * cn) / (1.0 + cn)
            return
            # TODO uncomment this if statement after U and MU computation modules are ready, check for correctness
            # if self.variables.COMPUTE_PHI_AND_T_PARTS:
            #     self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_all_roots_real(
            #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
            #         u3, self.variables.U_FINAL
            #     ) / math.sqrt(self.coefficient_list[1])

    def quartic_cases(self) -> None:
        self.variables['MU_INTEGRAL'] = self.compute_mu_integral(3)
        if self.variables.MU_INTEGRAL == -1.0:
            self.variables['U_FINAL'] = -1.0
            self.variables['CASE'] = 0
            self.variables['MU_TURNING_POINTS'] = 0
            return
        # now we search for U(u) solutions which will determine the case number
        list_of_exponents = [-1, -1, -1, -1, 0]
        if self.variables.COMPUTE_RELEVANT_VARIABLES:
            polynomial_coefficients = [
                complex(1.0, 0.0),
                complex(0.0, 0.0),
                complex(self.coefficient_list[0], 0.0),
                complex(self.coefficient_list[1], 0.0),
                complex(self.coefficient_list[2], 0.0)
            ]
            roots = polyroots.find_roots_of_polynomial(polynomial_coefficients)
            number_of_real_roots = 0
            for element in roots[1]:
                if element != 0.0:
                    continue
                else:
                    number_of_real_roots += 1
            if number_of_real_roots == 0:
                self.variables['CASE'] = 6
                self.variables['U_TURNING_POINTS'] = 0
            elif number_of_real_roots == 2:
                self.variables['CASE'] = 5
                self.variables['U_TURNING_POINTS'] = 0
                u1 = roots[0][0]
                if roots[1][1] == 0.0:
                    u4 = roots[0][1]
                else:
                    u4 = roots[0][3]
                self.variables['U14'] = [u1, 0.0, 0.0, u4]
            else:
                self.variables['U14'] = [element for element in roots[0]]

                def conditions_check() -> None:
                    if roots[0][1] > self.variables.U_PLUS and roots[0][2] > self.variables.U_PLUS:
                        self.variables['CASE'] = 5
                    elif self.variables.U_START <= roots[0][1]:
                        self.variables['CASE'] = 7
                    elif self.variables.U_START >= roots[0][2]:
                        self.variables['CASE'] = 8
                    else:
                        # unphysical real quartic
                        if self.variables.U_INTEGRAL_SIGN == 1.0:
                            self.variables['U_START'] = roots[0][2]
                        else:
                            self.variables['U_START'] = roots[0][1]

                conditions_check()
                if roots[0][1] > self.variables.U_PLUS and roots[0][2] > self.variables.U_PLUS:
                    self.variables['CASE'] = 5
                elif self.variables.U_START <= roots[0][1]:
                    self.variables['CASE'] = 7
                elif self.variables.U_START >= roots[0][2]:
                    self.variables['CASE'] = 8
        if self.variables.CASE == 5:
            # here we have quartic case with two real and two complex roots
            carter_constant_sign = np.sign(1.0, self.variables.CARTER_CONST)
            fgh_coefficients_for_elliptic_integral = [
                -1.0 * carter_constant_sign / np.abs(self.coefficient_list[2]) / self.variables.U14[0] /
                self.variables.U14[3],
                -1.0 * carter_constant_sign * (self.variables.U14[0] + self.variables.U14[3]) /
                np.abs(self.coefficient_list[2]) / self.variables.U14[0] / self.variables.U14[0] /
                self.variables.U14[3] / self.variables.U14[3],
                1.0
            ]
            if self.variables.U_INTEGRAL_SIGN > 0.0:
                self.variables['IU_T'] = ints.elliptical_integral_quartic_two_real_and_two_complex_roots(
                    p=list_of_exponents,
                    a=[-1.0 * self.variables.U14[0], 0.0, 0.0, carter_constant_sign * self.variables.U14[3], 0.0],
                    b=[1.0, 0.0, 0.0, -1.0 * carter_constant_sign, 0.0],
                    fgh=fgh_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=self.variables.U_START,
                    x=self.variables.U_PLUS
                ) / math.sqrt(np.abs(self.coefficient_list[2]))
            else:
                self.variables['IU_T'] = ints.elliptical_integral_quartic_two_real_and_two_complex_roots(
                    p=list_of_exponents,
                    a=[-1.0 * self.variables.U14[0], 0.0, 0.0, carter_constant_sign * self.variables.U14[3], 0.0],
                    b=[1.0, 0.0, 0.0, -1.0 * carter_constant_sign, 0.0],
                    fgh=fgh_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=0.0,
                    x=self.variables.U_START
                ) / math.sqrt(np.abs(self.coefficient_list[2]))
            if self.variables.MU_INTEGRAL > self.variables.IU_T:
                self.variables['U_FINAL'] = -1.0
                self.variables['CASE'] = 0
                self.variables['U_TURNING_POINTS'] = 0
                return
            if carter_constant_sign == 1.0:
                ub = self.variables.U14[0]
            else:
                ub = self.variables.U14[3]
            m = -0.5 * fgh_coefficients_for_elliptic_integral[1]
            n2 = fgh_coefficients_for_elliptic_integral[0] - fgh_coefficients_for_elliptic_integral[1] ** 2.0 / 4.0
            c4 = math.sqrt((m - self.variables.U14[3]) ** 2.0 + n2)
            c5 = math.sqrt((m - self.variables.U14[0]) ** 2.0 + n2)
            c1 = math.sqrt(np.abs(self.coefficient_list[2] * c4 * c5))
            if self.variables.COMPUTE_RELEVANT_VARIABLES:
                self.variables['IU0'] = ints.elliptical_integral_quartic_two_real_and_two_complex_roots(
                    p=list_of_exponents,
                    a=[-1.0 * self.variables.U14[0], 0.0, 0.0, carter_constant_sign * self.variables.U14[3], 0.0],
                    b=[1.0, 0.0, 0.0, -1.0 * carter_constant_sign, 0.0],
                    fgh=fgh_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=ub,
                    x=self.variables.U_START
                ) * self.variables.U_INTEGRAL_SIGN / math.sqrt(np.abs(self.coefficient_list[2]))
            m1 = carter_constant_sign * ((c4 + carter_constant_sign * c5) ** 2.0 -
                                         (self.variables.U14[3] - self.variables.U14[0]) ** 2.0) / 4.0 / c4 / c5
            jarg = c1 * (self.variables.IU0 + self.variables.MU_INTEGRAL)
            sn, cn, dn = scipy.special.ellipj(jarg, m1)
            numerator = self.variables.U14[3] * c5 + carter_constant_sign * self.variables.U14[0] - \
                        (carter_constant_sign * self.variables.U14[3] * c5 - self.variables.U14[0] * c4) * cn
            denominator = ((c4 - carter_constant_sign * c5) * cn + carter_constant_sign * c4 + c5)
            self.variables['U_FINAL'] = numerator / denominator
            # TODO uncomment this if statement after U and MU computation modules are ready, check for correctness
            # if self.variables.COMPUTE_PHI_AND_T_PARTS:
            #     self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_all_roots_real(
            #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
            #         u3, self.variables.U_FINAL
            #     ) / math.sqrt(self.coefficient_list[1])
            return
        elif self.variables.CASE == 6:
            # here we have quartic case with only complex roots
            # first thing we need to do it to find fgh coefficients
            polynomial_coefficients = [
                complex(1.0, 0.0),
                complex(-1.0 * self.coefficient_list[0] / math.sqrt(self.coefficient_list[2]), 0.0),
                complex(-1.0, 0.0),
                complex(math.sqrt(self.coefficient_list[2]) * (2.0 * self.coefficient_list[0] /
                                                               self.coefficient_list[2] - (self.coefficient_list[0] /
                                                                                           self.coefficient_list[
                                                                                               2]) ** 2.0), 0.0),
                complex(-1.0, 0.0),
                complex(-1.0 * self.coefficient_list[0] / math.sqrt(self.coefficient_list[2]), 0.0),
                complex(1.0, 0.0)
            ]
            roots = polyroots.find_roots_of_polynomial(polynomial_coefficients)
            h1 = 1.0
            for index, value in enumerate(roots[1]):
                if value == 0.0:
                    if roots[0][index] != 0.0:
                        h1 = roots[0][index]
                        break
            h2 = 1.0 / h1
            # g1 = self.coefficient_list[1] / self.coefficient_list[2] / (h2 - h1)
            # f1 = 1.0 / math.sqrt(self.coefficient_list[2])
            fgh1_coefficients_for_elliptic_integral = [
                1.0 / math.sqrt(self.coefficient_list[2]),
                self.coefficient_list[1] / self.coefficient_list[2] / (h2 - h1),
                h1
            ]
            fgh2_coefficients_for_elliptic_integral = [
                1.0 / math.sqrt(self.coefficient_list[2]),
                -1.0 * self.coefficient_list[1] / self.coefficient_list[2] / (h2 - h1),
                h2
            ]
            if self.variables.U_INTEGRAL_SIGN > 0.0:
                self.variables['IU_T'] = ints.elliptical_integral_quartic_all_complex_roots(
                    p=list_of_exponents,
                    a=[0.0, 0.0, 0.0, 0.0, 0.0],
                    b=[0.0, 0.0, 0.0, 0.0, 0.0],
                    fgh1=fgh1_coefficients_for_elliptic_integral,
                    fgh2=fgh2_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=self.variables.U_START,
                    x=self.variables.U_PLUS
                ) / math.sqrt(np.abs(self.coefficient_list[2]))
            else:
                self.variables['IU_T'] = ints.elliptical_integral_quartic_all_complex_roots(
                    p=list_of_exponents,
                    a=[0.0, 0.0, 0.0, 0.0, 0.0],
                    b=[0.0, 0.0, 0.0, 0.0, 0.0],
                    fgh1=fgh1_coefficients_for_elliptic_integral,
                    fgh2=fgh2_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=0.0,
                    x=self.variables.U_START
                ) / math.sqrt(np.abs(self.coefficient_list[2]))
            if self.variables.MU_INTEGRAL > self.variables.IU_T:
                self.variables['U_FINAL'] = -1.0
                self.variables['CASE'] = 0
                self.variables['U_TURNING_POINTS'] = 0
                return
            # Now we would like to have real and imaginary parts of the roots M plus/min and P plus/minus in terms
            # of real quantities
            polynomial_coefficients = [
                1.0 / self.coefficient_list[2] ** 3.0,
                -1.0 * self.coefficient_list[0] / self.coefficient_list[2] ** 3.0,
                -1.0 / self.coefficient_list[2] ** 2.0,
                -1.0 * (self.coefficient_list[1] ** 2.0 / self.coefficient_list[2] - 2.0 * self.coefficient_list[0])
                / self.coefficient_list[2] ** 2.0,
                -1.0 / self.coefficient_list[2],
                -1.0 * self.coefficient_list[0] / self.coefficient_list[2],
                1.0
            ]
            roots = polyroots.find_roots_of_polynomial(polynomial_coefficients)
            mn2 = 0.0
            for index, value in enumerate(roots[1]):
                if value == 0.0:
                    if roots[0][index] != 0.0:
                        mn2 = roots[0][index]
                        break
            p = self.coefficient_list[1] / (2.0 * self.coefficient_list[2] ** 2.0 *
                                            (mn2 ** 2.0 - 1.0 / self.coefficient_list[2]))
            m = -0.5 * self.coefficient_list[2] / self.coefficient_list[1] - p
            if m < p:
                temp = p
                p = m
                m = temp
            pr2 = 1.0 / mn2 / self.coefficient_list[2]
            n = math.sqrt(mn2 - m * m)
            r = math.sqrt(pr2 - p * p)
            c4 = math.sqrt((m - p) ** 2.0 + (n + r) ** 2.0)
            c5 = math.sqrt((m - p) ** 2.0 + (n - r) ** 2.0)
            c1 = 0.5 * (c4 + c5) * math.sqrt(np.abs(self.coefficient_list[2]))
            c2 = math.sqrt((4.0 * n * n - (c4 - c5) ** 2.0) / ((c4 + c5) ** 2.0 - 4.0 * n * n))
            c3 = m + c2 * n
            if self.variables.COMPUTE_RELEVANT_VARIABLES:
                self.variables['IU0'] = ints.elliptical_integral_quartic_all_complex_roots(
                    p=list_of_exponents,
                    a=[0.0, 0.0, 0.0, 0.0, 0.0],
                    b=[0.0, 0.0, 0.0, 0.0, 0.0],
                    fgh1=fgh1_coefficients_for_elliptic_integral,
                    fgh2=fgh2_coefficients_for_elliptic_integral,
                    ffr=self.variables.RF_IU_U1,
                    y=c3,
                    x=self.variables.U_START
                ) * np.sign(1.0, (self.variables.U_START - c3)) / math.sqrt(np.abs(self.coefficient_list[2]))
            m1 = ((c4 - c5) / (c4 + c5)) ** 2.0
            jarg = c1 * (self.variables.U_INTEGRAL_SIGN * self.variables.MU_INTEGRAL + self.variables.IU0)
            sn, cn, dn = scipy.special.ellipj(jarg, m1)
            sc = sn / cn
            self.variables['U_FINAL'] = c3 + (n * (1.0 + c2 ** 2.0) * sc) / (1.0 - c2 * sc)
            # TODO uncomment this if statement after U and MU computation modules are ready, check for correctness
            # if self.variables.COMPUTE_PHI_AND_T_PARTS:
            #     self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_all_roots_real(
            #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
            #         u3, self.variables.U_FINAL
            #     ) / math.sqrt(self.coefficient_list[1])
            return
        elif self.variables.CASE == 7:
            if self.variables.COMPUTE_RELEVANT_VARIABLES and self.variables.U_START != self.variables.U14[1]:
                self.variables['IU0'] = ints.elliptical_integral_quartic_all_real_roots(
                    p=list_of_exponents,
                    a=[-1.0 * self.variables.U14[0], self.variables.U14[1], self.variables.U14[2],
                       self.variables.U14[3], 0.0],
                    b=[1.0, -1.0, -1.0, -1.0, 0.0],
                    ffr=self.variables.RF_IU_U1,
                    y=self.variables.U_START,
                    x=self.variables.U14[1]
                )
            jarg = 0.5 * (self.variables.MU_INTEGRAL - self.variables.U_INTEGRAL_SIGN * self.variables.IU0 /
                          math.sqrt(-1.0 * self.coefficient_list[2])) * math.sqrt(
                np.abs(self.coefficient_list[2]) * (self.variables.U14[2] - self.variables.U14[0]) * (
                        self.variables.U14[3] - self.variables.U14[1])
            )
            m1 = ((self.variables.U14[3] - self.variables.U14[0]) *
                  (self.variables.U14[2] - self.variables.U14[1])) / \
                 (self.variables.U14[2] - self.variables.U14[0]) * (self.variables.U14[3] - self.variables.U14[1])
            sn, cn, dn = scipy.special.ellipj(jarg, m1)
            numerator = (self.variables.U14[1] - self.variables.U14[0]) * self.variables.U14[2] * sn * sn - \
                        self.variables.U14[1] * (self.variables.U14[2] - self.variables.U14[0])
            denominator = (self.variables.U14[1] - self.variables.U14[0]) * sn * sn - \
                          (self.variables.U14[2] - self.variables.U14[0])
            self.variables['U_FINAL'] = numerator / denominator
            self.variables['U_TURNING_POINTS'] = int(
                (1.0 - np.sign(1.0, self.variables.U_INTEGRAL_SIGN *
                               (self.variables.MU_INTEGRAL + self.variables.U_INTEGRAL_SIGN * self.variables.IU0
                                / math.sqrt(-1.0 * self.coefficient_list[2])))) / 2.0
            )
            # TODO uncomment this if statement after U and MU computation modules are ready, check for correctness
            # if self.variables.COMPUTE_PHI_AND_T_PARTS:
            #     self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_all_roots_real(
            #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
            #         u3, self.variables.U_FINAL
            #     ) / math.sqrt(self.coefficient_list[1])
            return
        else:
            if self.variables.COMPUTE_RELEVANT_VARIABLES and self.variables.U_START != self.variables.U14[2]:
                self.variables['IU0'] = ints.elliptical_integral_quartic_all_real_roots(
                    p=list_of_exponents,
                    a=[-1.0 * self.variables.U14[0], -1.0 * self.variables.U14[1], -1.0 * self.variables.U14[2],
                       self.variables.U14[3], 0.0],
                    b=[1.0, 1.0, 1.0, -1.0, 0.0],
                    ffr=self.variables.RF_IU_U1,
                    y=self.variables.U14[2],
                    x=self.variables.U_START
                )
            jarg = 0.5 * (self.variables.MU_INTEGRAL - self.variables.U_INTEGRAL_SIGN * self.variables.IU0 /
                          math.sqrt(-1.0 * self.coefficient_list[2])) * math.sqrt(
                np.abs(self.coefficient_list[2]) * (self.variables.U14[2] - self.variables.U14[0]) * (
                        self.variables.U14[3] - self.variables.U14[1])
            )
            m1 = ((self.variables.U14[3] - self.variables.U14[0]) *
                  (self.variables.U14[2] - self.variables.U14[1])) / \
                 (self.variables.U14[2] - self.variables.U14[0]) * (self.variables.U14[3] - self.variables.U14[1])
            sn, cn, dn = scipy.special.ellipj(jarg, m1)
            numerator = (self.variables.U14[3] - self.variables.U14[2]) * self.variables.U14[1] * sn * sn - \
                        self.variables.U14[2] * (self.variables.U14[3] - self.variables.U14[1])
            denominator = (self.variables.U14[3] - self.variables.U14[2]) * sn * sn - \
                          (self.variables.U14[3] - self.variables.U14[1])
            self.variables['U_FINAL'] = numerator / denominator
            self.variables['U_TURNING_POINTS'] = int(
                (1.0 - np.sign(1.0, self.variables.U_INTEGRAL_SIGN *
                               (self.variables.MU_INTEGRAL + self.variables.U_INTEGRAL_SIGN * self.variables.IU0
                                / math.sqrt(-1.0 * self.coefficient_list[2])))) / 2.0
            )
            # TODO uncomment this if statement after U and MU computation modules are ready, check for correctness
            # if self.variables.COMPUTE_PHI_AND_T_PARTS:
            #     self.variables['U_INTEGRAL'] = ints.elliptical_integral_cubic_all_roots_real(
            #         list_of_exponents, [-u1, -u2, -u3], [1.0, 1.0, 1.0], self.variables.RF_IU_U1,
            #         u3, self.variables.U_FINAL
            #     ) / math.sqrt(self.coefficient_list[1])
            return

    def compute(self) -> None:
        # we check for edge cases first
        self.edge_cases()
        # after basic edge cases we will solve equation U(u)
        # we have two cases here:
        # 1) U(u) is a third order polynomial
        # 2) U(u) is a fourth order polynomial
        # we start with U(u) as a third order polynomial
        # we are searching for roots of U(u) <=> solutions of equation U(u) = 0
        if self.coefficient_list[2] == 0.0 and self.coefficient_list[1] != 0:
            self.cubic_cases()
        # here we work with U(u) as a fourth order polynomial
        # we are searching for roots of U(u) <=> solutions of equation U(u) = 0
        else:
            self.quartic_cases()


if __name__ == '__main__':
    a = ComputeUFinal()
    print(a)
