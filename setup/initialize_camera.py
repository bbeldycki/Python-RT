import math
from setup.init_setup import Schema


def initialize_camera(input_data: Schema) -> None:
    input_data['BC'] = 'D'
    input_data.L = 1.00


class InitializeCamera:
    """
        In this class we initialize the camera using some default values form default input file as well as
        computing some necessary variables. To save us the trouble we will not use radius, we will use variable
        u = 1 / radius. This change takes care of possibility running into infinity by changes the computing range
        to [0, 1] interval. Moreover, we will use cosine of the inclination angle instead of the angle itself.
    """
    input_data: Schema

    def __init__(self, input_data: Schema):
        self.input_data = input_data
        self.offset = 0.5
        if self.input_data.NUMBER_OF_TRAJECTORIES == 1:
            self.offset = 1.0e-8

        self.max_ala_beta = max(self.input_data.ALFA_MIN ** 2.0, self.input_data.ALFA_MAX ** 2.0) ** 2.0 + \
                            max(self.input_data.BETA_MIN ** 2.0, self.input_data.BETA_MAX ** 2.0) ** 2.0

        self.u_start = min(1.0e-4, 1.0 / (self.input_data.OUTER_DISK_RADIUS * self.max_ala_beta))

        self.r_isco = self.innermost_stable_circular_orbit()
        self.u_isco = 1.0 / self.r_isco

        # event horizon size
        self.u_plus = 1.0 / (1.0 + math.sqrt(1.0 - self.input_data.SPIN ** 2.0))
        # u_minus should have the same form as u_plus with the sign difference of the denominator
        # but later in the code it will be very convenient to use inverse of u_minus
        # that is why we already define it that way
        self.u_minus = 1.0 - math.sqrt(1.0 - self.input_data.SPIN ** 2.0)

    def innermost_stable_circular_orbit(self) -> float:
        """
            This method computes the value of innermost stable circular orbit for rotating black hole, this means
            it needs the value of black hole spin in a certain units
        :return: value of innermost stable circular orbit (radius)
        """
        z1 = 1.0 + (1.0 - self.input_data.SPIN ** 2.0) ** (1.0 / 3.0) * (
                (1.0 + self.input_data.SPIN) ** (1.0 / 3.0) + (1.0 - self.input_data.SPIN) ** (1.0 / 3.0))
        z2 = (3.0 * self.input_data.SPIN ** 2.0 + z1 ** 2.0) ** 0.5
        return 3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)) ** 0.5


if __name__ == '__main__':
    camera = InitializeCamera()
