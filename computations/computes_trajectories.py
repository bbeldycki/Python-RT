class ComputeTrajectories:
    """
        Class responsible for computing the path of each trajectory whatever number of them may be.
        There are two ways of computing trajectory:
            1) we are looking for u_final, when we know u_start, mu_start and mu_final
            2) we are looking for mu_final, when we know u_start, u_final and mu_start
        Both ways requires different approach.
    """
    def __init__(self, camera: object):
        self.camera = camera

    def run_computations(self):
        pass


if __name__ == '__main__':
    cam = {}
    trajectories = ComputeTrajectories(cam)
    trajectories.run_computations()
