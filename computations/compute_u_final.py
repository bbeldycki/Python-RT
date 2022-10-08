import numpy as np


class ComputeUFinal:
    """
        Class responsible for computing u_final value when we know u_start, mu_start and mu_final
        Input variables will be passed in form of dict()
    """
    def __init__(self, input_variables: dict):
        self.input_variables = input_variables

        # set the coefficients for quadratic equation U(u)
        self.c_coefficient = self.input_variables['spin'] ** 2.0 - self.input_variables['angular_momentum'] ** 2.0 \
                             - self.input_variables['carter_c']
        self.d_coefficient = 2.0 * ((self.input_variables['spin'] - self.input_variables['angular_momentum']) ** 2.0
                                    + self.input_variables['carter_c'])
        self.e_coefficient = -1.0 * self.input_variables['spin'] ** 2.0 * self.input_variables['carter_c']

    #TODO ponizsze metody trzeba przepisac inaczej teraz juz rozumiem jak, tam moze byc 1 special case, ale na ogol bedzie
    #TODO tam liczona wartosc calki cmu i bedzie przekazywana dalej
    #TODO trzeba to rozbic na cos w stylu compute cmu i tutaj bedzie mozna zrobic zwortke special case

    def check_special_cases(self) -> list:
        if self.input_variables['carter_c'] == 0.0:
            return [True, 0]
        if self.input_variables['spin'] == 0.0:
            return [True, 1]
        return [False, None]

    def analytical_solutions(self, case: int) -> list:
        if case:
            # case for a = 0.0
            pass
        if self.input_variables['carter_c'] ** 2.0 >= self.input_variables['spin'] ** 2.0 or \
                self.input_variables['mu_start'] == 0.0:
            # we return following values
            # index 0 - value of mu (angular) integral
            # index 1 - value of u_final
            # index 2 - case number
            return [0.0, -1.0, 0]
        else:
            sign = np.sign(self.input_variables['mu_start'])
            a = sign * self.input_variables['sign_of_mu_integral']
            b = a * (-1.0) ** (self.input_variables['mu_turning_points'] + 1)
            mu_plus = sign * np.sqrt(1.0 - self.input_variables['angular_momentum'] ** 2.0 /
                                     self.input_variables['spin'] ** 2.0)

            pass

    def compute(self):
        pass


if __name__ == '__main__':
    a = ComputeUFinal()
    print(a)
