import math
import numpy as np
from integrals.integrals import rf


def mu_integral_symmetric_case_involving_mu_start(spin: float, mu_negative: float, mu_positive: float, mu_plus: float,
                                                  initial_mu: float) -> tuple[float, float, float, float]:
    """
            Function which computes MU(mu) integral pieces involving MU_START and MU turning points for the symmetric
            case which means M_{-} < 0.0.
            This function takes a following input variables:
            spin: black hole spin
            mu_negative, mu_positive: roots of MU(mu) equation in mu^{2} terms, in increasing order
            mu_plus: upper physical turning point
            initial_mu: value of MU_START
    """
    if mu_negative >= 0.0:
        raise Exception('Error occurred in mu_integral_symmetric_case_involving_mu_start. '
                        'We have situation where mu_negative is not lower than 0.0. '
                        'The value of mu_negative must be lower than 0.0')
    if mu_positive < mu_negative:
        raise Exception('Error occurred in mu_integral_symmetric_case_involving_mu_start. '
                        'We have situation where mu_positive is lower than mu_negative. '
                        'The value of mu_positive must be greater than mu_negative.')
    if mu_positive - mu_negative < 0.0:
        raise Exception('Error occurred in mu_integral_symmetric_case_involving_mu_start. '
                        'We have situation where mu_positive - mu_negative is lower than 0.0. '
                        'The value of mu_positive must be mu_positive - mu_negative must be non negative.')
    f1 = np.abs(spin) * math.sqrt(mu_positive - mu_negative)
    m1 = -1.0 * mu_negative / (mu_positive - mu_negative)
    phi0 = math.acos(initial_mu / mu_plus)
    s0 = math.sin(phi0)
    ak = math.sqrt(1.0 - m1)
    q0 = 1.0 - s0 * s0 * ak * ak  # (1.0 - s0 * ak) * (1.0 + s0 * ak)
    rf0 = rf(x=1.0 - s0 * s0, y=q0, z=1.0)
    rfc = rf(x=0.0, y=m1, z=1.0)

    if math.tan(phi0) < 0.0:
        imu0 = (2.0 * rfc - s0 * rf0) / f1
    else:
        imu0 = s0 * rf0 / f1
    imum = 2.0 * rfc / f1

    # old part of code, I left it just in case have to review function for any reason
    # imum = 2.0 * rfc
    # imu0 = s0 * rf0
    # if math.tan(phi0) < 0.0:
    #     imu0 = imum - imu0
    # imu0 = imu0 / f1
    # imum = imum / f1

    return rf0, rfc, imu0, imum


def mu_integral_symmetric_case_involving_mu_final(spin: float, mu_negative: float, mu_positive: float, mu_plus: float,
                                                  final_mu: float, imum: float) -> tuple[float, float]:
    """
            Function which computes MU(mu) integral pieces involving MU_FINAL for the symmetric case which
            means M_{-} < 0.0.
            This function takes a following input variables:
            spin: black hole spin
            mu_negative, mu_positive: roots of MU(mu) equation in mu^{2} terms, in increasing order
            mu_plus: upper physical turning point
            final_mu: value of MU_FINAL
            imum: value of MU integral between mu_minus and mu_plus
    """
    if mu_negative >= 0.0:
        raise Exception('Error occurred in mu_integral_symmetric_case_involving_mu_start. '
                        'We have situation where mu_negative is not lower than 0.0. '
                        'The value of mu_negative must be lower than 0.0')
    if mu_positive < mu_negative:
        raise Exception('Error occurred in mu_integral_symmetric_case_involving_mu_final. '
                        'We have situation where mu_positive is lower than mu_negative. '
                        'The value of mu_positive must be greater than mu_negative.')
    if mu_positive - mu_negative < 0.0:
        raise Exception('Error occurred in mu_integral_symmetric_case_involving_mu_final. '
                        'We have situation where mu_positive - mu_negative is lower than 0.0. '
                        'The value of mu_positive must be mu_positive - mu_negative must be non negative.')
    f1 = np.abs(spin) * math.sqrt(mu_positive - mu_negative)
    m1 = -1.0 * mu_negative / (mu_positive - mu_negative)
    phif = math.acos(final_mu / mu_plus)
    sf = math.sin(phif)
    ak = math.sqrt(1.0 - m1)
    qf = 1.0 - sf * sf * ak * ak  # (1.0 - sf * ak) * (1.0 + sf * ak)
    rff = rf(x=1.0 - sf * sf, y=qf, z=1.0)

    if math.tan(phif) < 0.0:
        imuf = (imum - sf * rff) / f1
    else:
        imuf = sf * rff / f1

    # old part of code, I left it just in case have to review function for any reason
    # imuf = s0 * rff
    # if math.tan(phi0) < 0.0:
    #     imuf = imum * f1 - imuf
    # imuf = imuf / f1

    return rff, imuf


def mu_integral_asymmetric_case_involving_mu_start(spin: float, mu_negative: float, mu_positive: float, mu_plus: float,
                                                   initial_mu: float) -> tuple[float, float, float, float]:
    """
               Function which computes MU(mu) integral pieces involving MU_START and MU turning points for the
               asymmetric case which means M_{-} > 0.0.
               This function takes a following input variables:
               spin: black hole spin
               mu_negative, mu_positive: roots of MU(mu) equation in mu^{2} terms, in increasing order
               mu_plus: upper physical turning point
               initial_mu: value of MU_START
       """
    if mu_positive < mu_negative:
        raise Exception('Error occurred in function mu_integral_asymmetric_case_involving_mu_start. '
                        'We have situation where mu_positive is lower than mu_negative. '
                        'The value of mu_positive must be greater than mu_negative.')
    if mu_positive - mu_negative < 0.0:
        raise Exception('Error occurred in mu_integral_asymmetric_case_involving_mu_start. '
                        'We have situation where mu_positive - mu_negative is lower than 0.0. '
                        'The value of mu_positive must be mu_positive - mu_negative must be non negative.')
    f1 = np.abs(spin) * mu_plus
    m1 = mu_negative / mu_positive
    s0 = math.sqrt((mu_positive - initial_mu * initial_mu) / (mu_positive - mu_negative))
    phi0 = math.asin(s0)
    ak = math.sqrt(1.0 - m1)
    q0 = 1.0 - s0 * s0 * ak * ak  # (1.0 - s0 * ak) * (1.0 + s0 * ak)
    rf0 = rf(x=1.0 - s0 * s0, y=q0, z=1.0)
    rfc = rf(x=0.0, y=m1, z=1.0)

    if math.tan(phi0) < 0.0:
        imu0 = (2.0 * rfc - s0 * rf0) / f1
    else:
        imu0 = s0 * rf0 / f1
    imum = rfc / f1

    # old part of code, I left it just in case have to review function for any reason
    # imum = 2.0 * rfc
    # imu0 = s0 * rf0
    # if math.tan(phi0) < 0.0:
    #     imu0 = imum - imu0
    # imu0 = imu0 / f1
    # imum = imum / f1 / 2.0

    return rf0, rfc, imu0, imum


def mu_integral_asymmetric_case_involving_mu_final(spin: float, mu_negative: float, mu_positive: float, mu_plus: float,
                                                   final_mu: float) -> tuple[float, float]:
    """
               Function which computes MU(mu) integral pieces involving MU_FINAL for the asymmetric case which
               means M_{-} > 0.0.
               This function takes a following input variables:
               spin: black hole spin
               mu_negative, mu_positive: roots of MU(mu) equation in mu^{2} terms, in increasing order
               mu_plus: upper physical turning point
               final_mu: value of MU_START
       """
    if mu_positive < mu_negative:
        raise Exception('Error occurred in function mu_integral_asymmetric_case_involving_mu_final. '
                        'We have situation where mu_positive is lower than mu_negative. '
                        'The value of mu_positive must be greater than mu_negative.')
    if mu_positive - mu_negative < 0.0:
        raise Exception('Error occurred in function mu_integral_asymmetric_case_involving_mu_final. '
                        'We have situation where mu_positive - mu_negative is lower than 0.0. '
                        'The value of mu_positive must be mu_positive - mu_negative must be non negative.')
    f1 = np.abs(spin) * mu_plus
    m1 = mu_negative / mu_positive
    sf = math.sqrt((mu_positive - final_mu * final_mu) / (mu_positive - mu_negative))
    ak = math.sqrt(1.0 - m1)
    qf = 1.0 - sf * sf * ak * ak  # (1.0 - sf * ak) * (1.0 + sf * ak)
    imuf = rf(x=1.0 - sf * sf, y=qf, z=1.0) * sf / f1

    # old part of code, I left it just in case have to review function for any reason
    # rff = rf(x=1.0 - sf * sf, y=qf, z=1.0)
    # imuf = sf * rff

    return rf(x=1.0 - sf * sf, y=qf, z=1.0), imuf
