import math


class VariablesSchema:
    SPIN: float = 0.998
    INITIAL_POLAR_ANGLE: int = 60  # initial polar angle in degrees
    COMPUTE_U_FINAL: bool = False
    COMPUTE_PHI_AND_T_PARTS: bool = False
    # this boolean variable force computation of various integrals otherwise they must be supplemented from input
    COMPUTE_RELEVANT_VARIABLES: bool = True
    NUMBER_OF_POINTS_ALONG_TRAJECTORY: int = 500
    U_START: float = 0.0  # initial U translates to R
    MU_START: float = math.cos(INITIAL_POLAR_ANGLE * math.pi / 180.0)  # initial MU translate to initial polar angle
    U_FINAL: float = 0.0  # final (computed/given) U
    MU_FINAL: float = 0.0  # final (computed/given) MU
    L: float = 0.0  # angular momentum
    CARTER_CONST: float = 0.0  # Carter constant
    U_INTEGRAL_SIGN: float = 1.0  # sign of U integral
    MU_INTEGRAL_SIGN: float = 1.0  # sign of MU integral
    U_TURNING_POINTS: int = 0  # number of turning points along U integral
    MU_TURNING_POINTS: int = 0  # number of turning points along MU integral
    U_PLUS: float = 1.0 / (1.0 + math.sqrt(1.0 - SPIN * SPIN))

    # computational variables
    CASE: int = 0  # case number, computed if COMPUTE_RELEVANT_VARIABLES
    U14: list[float] = [0, 0, 0, 0]  # U1, U2, U3, U4 - real roots of U(u), computed if COMPUTE_RELEVANT_VARIABLES
    H1: float = 0.0  # computed if COMPUTE_RELEVANT_VARIABLES
    MU_INTEGRAL: float = 0.0  # value of MU integral between MU_START and MU_FINAL
    U_INTEGRAL: float = 0.0  # value of U integral between U_START and U_FINAL
    # value of R-Functions relevant for U_START piece of U integral, computed if COMPUTE_RELEVANT_VARIABLES
    RF_IU_U0: float = 0.0
    # value of R-Functions relevant for U_FINAL piece of U integral, computed if COMPUTE_RELEVANT_VARIABLES
    RF_IU_U1: float = 0.0
    # value of R-Functions relevant for MU_START piece of MU integral, computed if COMPUTE_RELEVANT_VARIABLES
    RF_IMU_MU1: float = 0.0
    # value of R-Functions relevant for MU_FINAL piece of MU integral, computed if COMPUTE_RELEVANT_VARIABLES
    RF_IMU_MU2: float = 0.0
    # value of R-Functions relevant for turning point piece of MU integral, computed if COMPUTE_RELEVANT_VARIABLES
    RF_IMU_MU3: float = 0.0
    # value of U integral between U_START and relevant turning points if exists, computed if COMPUTE_RELEVANT_VARIABLES
    # if there is no turning point present then it set to 0
    IU_U0_T: float = 0.0
    IU_T_U1: float = 0.0
    IU_T: float = 0.0
    IU: float = 0.0
    IU0: float = 0.0
    # value of MU integral between MU_START and MU_PLUS, computed if COMPUTE_RELEVANT_VARIABLES
    IMU_MU0_MUPLUS: float = 0.0
    # value of MU integral between MU_MINUS and MU_PLUS, computed if COMPUTE_RELEVANT_VARIABLES
    IMU_MUMINUS_MUPLUS: float = 0.0

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def __getitem__(self, key):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return None


variables: VariablesSchema = VariablesSchema()