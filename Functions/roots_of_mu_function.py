import numpy as np


def roots_of_mu_function(black_hole_spin: float, cos_of_inclination_angle: float, carter_constant: float,
                         angular_momentum: float) -> list:

    temporary = black_hole_spin ** 2 - carter_constant - angular_momentum ** 2
    solution = - 0.5 * (temporary + np.sign(temporary) * np.sqrt(temporary ** 2 +
                                                                 4.0 * carter_constant * black_hole_spin ** 2))
    if temporary < 0.0:
        m_minus = -solution / black_hole_spin ** 2
        m_plus = carter_constant / solution
    else:
        m_minus = carter_constant / solution
        m_plus = -solution / black_hole_spin ** 2
    m_plus = np.minimum(m_plus, 1.0)
    if m_minus > 0.0:
        # case with asymmetric roots
        miu_plus = np.sign(cos_of_inclination_angle) * np.sqrt(m_plus)
        miu_minus = np.sign(cos_of_inclination_angle) * np.sqrt(m_minus)
        return [cos_of_inclination_angle if np.abs(miu_plus) > np.abs(cos_of_inclination_angle) else miu_plus,
                cos_of_inclination_angle if np.abs(miu_minus) > np.abs(cos_of_inclination_angle) else miu_minus]
    else:
        # case with symmetric roots. In this case the orbit crosses the equatorial plane
        miu_plus = np.sqrt(m_plus)
        miu_plus = cos_of_inclination_angle if np.abs(miu_plus) < np.abs(cos_of_inclination_angle) else miu_plus
        return [miu_plus, -miu_plus]


if __name__ == '__main__':
    a = 0.998
    q2 = -111.0
    ll = -12.5
    mup = np.cos(89 * np.pi / 180)
    roots = roots_of_mu_function(black_hole_spin=a, cos_of_inclination_angle=mup, carter_constant=q2,
                                 angular_momentum=ll)
    print(mup, sorted(roots))
