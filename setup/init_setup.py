import math
from enum import Enum


class GridType(Enum):
    SQUARE = 'square'
    CIRCLE = 'circle'


class Schema:
    SPIN: float = 0.998
    INITIAL_POLAR_ANGLE = 60  # initial polar angle in degrees
    NUMBER_OF_TRAJECTORIES = 1
    COMPUTE_U_FINAL = False
    COMPUTE_PHI_AND_T_PARTS = False
    SELECT_GRID_TYPE: GridType = GridType.SQUARE
    ALFA_MIN = -10.0
    ALFA_MAX = 10.0
    BETA_MIN = -10.0
    BETA_MAX = 10.0
    NUMBER_OF_POINTS_ALONG_X_AXIS = 20
    NUMBER_OF_POINTS_ALONG_Y_AXIS = 20
    NUMBER_OF_POINTS_ALONG_TRAJECTORY = 500
    OUTER_DISK_RADIUS = 100.0
    U_START = 0.0
    MU_START = math.cos(INITIAL_POLAR_ANGLE * math.pi / 180.0)
    U_FINAL = 0.0
    MU_FINAL = 0.0
    L = None
    CARTER_CONST = 0.0
    U_INTEGRAL_SIGN = 1.0
    MU_INTEGRAL_SIGN = 1.0
    U_TURNING_POINTS = 0
    MU_TURNING_POINTS = 0

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def __getitem__(self, key):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return None


input_variables: Schema = Schema()
