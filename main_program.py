from setup.init_setup import input_variables
from setup.initialize_camera import initialize_camera


def compute():
    initialize_camera(input_variables)
    for i in range(input_variables.NUMBER_OF_TRAJECTORIES):
        print(i)


if __name__ == '__main__':
    compute()
