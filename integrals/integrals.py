import math


def compute_a_p1_pn(p: list, a: list, b: list, y: float, x: float) -> float:
    """
        Function which computed value of coefficient A from equation 2.6 from different Carlson Papers
        B. C; Carlson; A Table of Elliptic Integrals: Cubic Cases
        Mathematics of Computation, Volume 53, Number 187, Pages 327-333, July 1989
        AND
        B.C; Carlson; A Table of Elliptic Integrals of the Third Kind
        Mathematics of Computation, Volume 51, Number 183, Pages 267-280, July 1988
    """
    if x < y:
        raise Exception('Error occurred in function compute_a_p1_pn. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.2
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    ax = 1
    ay = 1
    # coefficients A from equation 2.6
    for j in range(len(a)):
        ax = ax * xi[j] ** p[j]
        ay = ay * yi[j] ** p[j]
    return ax - ay


def compute_a_p1_pn_double_quadratic(p: list, a: list, b: list, fgh1: list, fgh2: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function compute_a_p1_pn_double_quadratic. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    fi = [fgh1[0], fgh2[0]]
    gi = [fgh1[1], fgh2[1]]
    hi = [fgh1[2], fgh2[2]]
    ksi = []
    eta = []
    # coefficient ksi and eta from equation 2.1
    for i in range(len(fi)):
        ksi.append(math.sqrt(fi[i] + gi[i] * x + hi[i] * x * x))
        eta.append(math.sqrt(fi[i] + gi[i] * y + hi[i] * y * y))
    ax = 1
    ay = 1
    if p[4] != 0:
        ksi5 = a[4] + b[4] * x
        eta5 = a[4] + b[4] * y
        for i in range(len(p)):
            if i == 2 or i == 3:
                continue
            elif i == 4:
                ax = ax * math.sqrt(ksi5 ** p[i])
                ay = ay * math.sqrt(eta5 ** p[i])
            else:
                ax = ax * ksi[i] ** p[i]
                ay = ay * eta[i] ** p[i]
        return ax - ay
    else:
        for i in range(2):
            ax = ax * ksi[i] ** p[i]
            ay = ay * eta[i] ** p[i]
        return ax - ay


def compute_a_p1_pn_quadratics(p: list, a: list, b: list, fgh: list, y: float, x: float) -> float:
    """
        Function which computed value of coefficient A from equation 2.6 from different Carlson Papers
        B.C; Carlson; A Table of Elliptic Integrals: One Quadratic factor
        Mathematics of Computation, Volume 56, Number 183, Pages 267-280, July 1991
    """
    if x < y:
        raise Exception('Error occurred in function compute_a_p1_pn_quadratics. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.2
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    # coefficients ksi and eta given from equation 2.4
    ksi = math.sqrt(fgh[0] + fgh[1] * x + fgh[2] * x * x)
    eta = math.sqrt(fgh[0] + fgh[1] * y + fgh[2] * y * y)
    ax = 1
    ay = 1
    # coefficients A from equation 2.5
    for j in range(len(a)):
        if j != 1:
            if j == 2:
                continue
            ax = ax * xi[j] ** p[j]
            ay = ay * yi[j] ** p[j]
        else:
            ax = ax * ksi ** p[j]
            ay = ay * eta ** p[j]
    return ax - ay


def compute_alfa_beta_i5(f: list, g: list, h: list, a5: float, b5: float, k: int) -> tuple:
    """
        Function which computes coefficients alfa_i5 and beta_i5
    """

    return 2 * f[k] * b5 - g[k] * a5, g[k] * b5 - 2 * h[k] * a5


def compute_cij_value(a: list, b: list, fgh: list, k: int, k1: int) -> float:
    return 2 * b[k] * b[k1] * fgh[0] - fgh[1] * (a[k] * b[k1] + a[k1] * b[k]) + 2 * fgh[2] * a[k] * a[k1]


def compute_delta_ij(f: list, g: list, h: list, k: int, m: int) -> float:
    """
        Function which computes coefficients delta_ij
    """
    return 2 * f[k] * h[m] + 2 * f[m] * h[k] - g[k] * g[m]


def compute_gamma_i5(f: list, g: list, h: list, a5: float, b5: float, k: int) -> float:
    """
        Function which computes coefficients gamma_i5
    """
    #
    gama = f[k] * b5 * b5 - g[k] * a5 * b5 + h[k] * a5 * a5
    if gama > 0.0:
        return gama
    else:
        raise Exception('Error occurred during comparison of gamma_i5 in function compute_gamma_i5. '
                        'Resulting value was negative or zero,but should be greater than zero ')


def iteration_loop_for_rc_function(x: float, y: float) -> tuple:
    """
            Function which contains all calculations needed to be done in iteration
    """
    alamb = 2.0 * math.sqrt(x) * math.sqrt(y) + y
    x = 0.25 * (x + alamb)
    y = 0.25 * (y + alamb)
    ave = (x + 2 * y) / 3.0
    s = (y - ave) / ave
    return x, y, ave, s


def iteration_loop_for_rd_function(x: float, y: float, z: float, s: float, f: float) -> tuple:
    """
        Function which contains all calculations needed to be done in iteration
    """
    alamb = math.sqrt(x) * (math.sqrt(y) + math.sqrt(z)) + math.sqrt(y) * math.sqrt(z)
    s = s + f / (math.sqrt(z) * (z + alamb))
    f = 0.25 * f
    x = 0.25 * (x + alamb)
    y = 0.25 * (y + alamb)
    z = 0.25 * (z + alamb)
    ave = 0.2 * (x + y + 3.0 * z)
    delx = (ave - x) / ave
    dely = (ave - y) / ave
    delz = (ave - z) / ave
    return x, y, z, delx, dely, delz, ave, s, f


def iteration_loop_for_rf_function(x: float, y: float, z: float) -> tuple:
    """
        Function which contains all calculations needed to be done in iteration
    """
    alamb = math.sqrt(x) * (math.sqrt(y) + math.sqrt(z)) + math.sqrt(y) * math.sqrt(z)
    x = 0.25 * (x + alamb)
    y = 0.25 * (y + alamb)
    z = 0.25 * (z + alamb)
    ave = (x + y + z) / 3.0
    delx = (ave - x) / ave
    dely = (ave - y) / ave
    delz = (ave - z) / ave
    return x, y, z, delx, dely, delz, ave


def iteration_loop_for_rj_function(x: float, y: float, z: float, p: float, s: float, f: float) -> tuple:
    """
        Function which contains all calculations needed to be done in iteration
    """
    alamb = math.sqrt(x) * (math.sqrt(y) + math.sqrt(z)) + math.sqrt(y) * math.sqrt(z)
    alpha = (p * (math.sqrt(x) + math.sqrt(y) + math.sqrt(z)) + math.sqrt(x) * math.sqrt(y) * math.sqrt(z)) ** 2
    beta = p * (p + alamb) ** 2
    s = s + f * rc(alpha, beta)
    f = 0.25 * f
    x = 0.25 * (x + alamb)
    y = 0.25 * (y + alamb)
    z = 0.25 * (z + alamb)
    p = 0.25 * (p + alamb)
    ave = 0.2 * (x + y + z + 2 * p)
    delx = (ave - x) / ave
    dely = (ave - y) / ave
    delz = (ave - z) / ave
    delp = (ave - p) / ave
    return x, y, z, p, delx, dely, delz, delp, ave, s, f


def rc(x: float, y: float) -> float:
    """
        Function to calculate elliptical integral of the third kind RC
        The integral takes the following form
        RC(x, y) = 1/2 int from 0 do infinity dt (t + x)^{-1/2} * (t + y)^{-1}
        Value x must be greater than 0
        Value of y must be nonnegative
    """

    if x < 0:
        raise Exception('Error occurred in function rc. x must be greater than 0')
    elif y == 0.0:
        raise Exception('Error occurred in function rc. y cannot be equal to 0.')
    elif x + abs(y) < 1.69e-38:
        raise Exception('Error occurred in function rc. Both x and y are to close to 0.')
    elif x + abs(y) > 3e37:
        raise Exception('Error occurred in function rc. Sum of x and module of y is to large.')
    elif y < -1.72e19 and 0.0 < x < 0.01:
        raise Exception('Error occurred in function rc. Sum of y is a large negative number and x is a small number.')
    if y > 0.0:
        w = 1.0
        x_new = x
        y_new = y
    else:
        x_new = x - y
        y_new = -1 * y
        w = math.sqrt(x) / math.sqrt(x_new)
    # First iteration
    x_new, y_new, ave, s = iteration_loop_for_rc_function(x_new, y_new)
    # Next iterations if needed
    while abs(s) > 0.04:
        x_new, y_new, ave, s = iteration_loop_for_rc_function(x_new, y_new)
    # Final integral value
    integral_value = 0.375 + 9.0 * s / 22.0
    integral_value = 1.0 / 7.0 + s * integral_value
    integral_value = 0.3 + s * integral_value
    integral_value = 1.0 + s * s * integral_value
    integral_value = w * integral_value / math.sqrt(ave)
    return integral_value


def rd(x: float, y: float, z: float) -> float:
    """
        Function to calculate elliptical integral of the second kind RD
        The integral takes the following form
        RD(x, y, z) = 3/2 int from 0 do infinity dt [(t + x) * (t + y)]^{-1/2} * (t + z)^{-3/2}
        Values x, y must be greater than 0 and only of them can be equal to 0
        Value of z must be positive
    """

    if min(x, y) < 0:
        raise Exception('Error occurred in function rd. One or more arguments was smaller than 0')
    elif min(x + y, z) < 1.5e-25:
        raise Exception('Error occurred in function rd. Sum of pair (x + y) or value of z was to close to 0.')
    elif max(x, y, z) > 4.5e25:
        raise Exception('Error occurred in function rd. One of the arguments was to large.')
    sums = 0.0
    fac = 1.0
    # First iteration
    x_temp, y_temp, z_temp, delx, dely, delz, ave, sums, fac = iteration_loop_for_rd_function(x, y, z, sums, fac)
    # Next iterations if needed
    while max(abs(delx), abs(dely), abs(delz)) > 0.05:
        x_temp, y_temp, z_temp, delx, dely, delz, ave, sums, fac = iteration_loop_for_rd_function(x_temp, y_temp,
                                                                                                  z_temp, sums, fac)
    # We use the following coefficients to compute final integral value
    coef_e2 = delx * dely
    coef_e3 = coef_e2 - delz * delz
    coef_e4 = coef_e2 - 6.0 * delz * delz
    coef_e5 = coef_e4 + 2.0 * coef_e3
    # Final integral value
    integral_value = -9 * coef_e3 / 22.0 + 3.0 * delz * coef_e2 / 26.0
    integral_value = delz * integral_value + coef_e5 / 6.0
    integral_value = delz * integral_value + 1.0
    integral_value = integral_value + coef_e4 * (-3.0 / 14.0 + 9.0 * coef_e4
                                                 - 45.0 * coef_e5 * delz / 260.0)
    integral_value = fac * integral_value
    integral_value = 3.0 * sums + integral_value / (ave * math.sqrt(ave))
    return integral_value


def rf(x: float, y: float, z: float) -> float:
    """
        Function to calculate elliptical integral of the first kind RF
        The integral takes the following form
        RF(x, y, z) = 1/2 int from 0 do infinity dt [(t + x) * (t + y) * (t + z)]^{-1/2}
        Values x, y and z must be greater than 0 and only of them can be equal to 0
    """

    if min(x, y, z) < 0.0:
        raise Exception('Error occurred in function rf. One or more arguments was smaller than 0')
    elif min(x + y, x + z, y + z) < 1.5e-38:
        raise Exception('Error occurred in function rf. Sum on one pair of numbers ex. (x + y) was to close to 0.')
    elif max(x, y, z) > 3e37:
        raise Exception('Error occurred in function rf. One of the arguments was to large.')
    # First iteration
    x_temp, y_temp, z_temp, delx, dely, delz, ave = iteration_loop_for_rf_function(x, y, z)
    # Next iterations if needed
    while max(abs(delx), abs(dely), abs(delz)) > 0.08:
        x_temp, y_temp, z_temp, delx, dely, delz, ave = iteration_loop_for_rf_function(x_temp, y_temp, z_temp)
    # We use the following coefficients to compute final integral value
    coef_e2 = delx * dely - delz * delz
    coef_e3 = delx * dely * delz
    # Final integral value
    integral_value = (1.0 + (coef_e2 / 24.0 - 0.1 - 3.0 * coef_e3 / 44.0) *
                      coef_e2 + coef_e3 / 14.0) / math.sqrt(ave)
    return integral_value


def rj(x: float, y: float, z: float, p: float) -> float:
    """
        Function to calculate elliptical integral of the third kind RJ
        The integral takes the following form
        RJ(x, y, z, p) = 3/2 int from 0 do infinity dt [(t + x) * (t + y) * (t + z)]^{-1/2} * (t + p)^{-1}
        Values x, y and z must be greater than 0 and at most one can by equal 0
        Value of p must be nonzero, if p < 0 then the Cauchy principal value is returned
    """

    if min(x, y, z) < 0.0:
        raise Exception('Error occurred in function rj. The lowest of arguments x, y or z is lower than 0.')
    elif min(x + y, x + z, y + z, abs(p)) < 2.5e-13:
        raise Exception('Error occurred in function rj. Sum of pair x + y or x + z or y + z or '
                        'absolute value of p is to close to 0.')
    elif max(x, y, z, abs(p)) > 9e11:
        raise Exception('Error occurred in function rj. x or y or z or absolute of p is to large.')
    sums = 0.0
    fac = 1.0
    if p > 0.0:
        x_new = x
        y_new = y
        z_new = z
        pt = p
    elif p < 0.0:
        x_new = min(x, y, z)
        z_new = max(x, y, z)
        y_new = x + y + z - x_new - z_new
        a1 = 1.0 / (y_new - p)
        b1 = a1 * (z_new - y_new) * (y_new - x_new)
        pt = y_new + b1
        rho = x_new * z_new / y_new
        tau = p * pt / y_new
        rcx = rc(rho, tau)
    # First iteration
    x_new, y_new, z_new, pt, delx, dely, delz, delp, ave, sums, fac = iteration_loop_for_rj_function(
        x_new, y_new, z_new, pt, sums, fac)
    # Next iterations if needed
    while max(abs(delx), abs(dely), abs(delz), abs(delp)) > 0.05:
        x_new, y_new, z_new, pt, delx, dely, delz, delp, ave, sums, fac = iteration_loop_for_rj_function(
            x_new, y_new, z_new, pt, sums, fac)
    # We use the following coefficients to compute final integral value
    coef_e2 = delx * (dely + delz) + dely * delz
    coef_e3 = delx * dely * delz
    coef_e4 = delp * delp
    coef_e5 = coef_e2 - 3.0 * coef_e4
    coef_e6 = coef_e3 + 2.0 * delp * (coef_e2 - coef_e4)
    # Final integral value
    integral_value = 3.0 * delp / 26.0 - 3.0 / 11.0
    integral_value = (1.0 / 6.0 + delp * integral_value) * coef_e3
    integral_value = integral_value - delp * coef_e4 / 3.0
    integral_value = delp * coef_e2 * (1.0 / 3.0 - 3.0 * delp / 22.0) + integral_value
    integral_value = 1.0 + coef_e5 * (9.0 * coef_e5 - 4.5 * coef_e6 / 26.0 - 3.0 / 14.0) + integral_value
    integral_value = fac * integral_value
    integral_value = 3.0 * sums + integral_value / (ave * math.sqrt(ave))
    if p <= 0.0:
        integral_value = a1 * (b1 * integral_value + 3.0 * (rcx - rf(x_new, y_new, z_new)))
    return integral_value


def elliptical_integral_cubic_all_roots_real(p: list[int], a: list[float], b: list[float], ffr: float, y: float,
                                             x: float) -> float:
    """
        Integral computed based on numerical method by B. C; Carlson; A Table of Elliptic Integrals: Cubic Cases
        Mathematics of Computation, Volume 53, Number 187, Pages 327-333, July 1989
    """

    if x < y:
        raise Exception('Error occurred in function elliptical_integral_cubic_all_roots_real. We have situation where '
                        'x is lower than y. The value of x must be greater than y')
    # coefficients d_{i,j} from equation 2.1
    d12 = a[0] * b[1] - a[1] * b[0]
    d13 = a[0] * b[2] - a[2] * b[0]
    d14 = a[0] * b[3] - a[3] * b[0]
    d24 = a[1] * b[3] - a[3] * b[1]
    d34 = a[2] * b[3] - a[3] * b[2]
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.2
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    # coefficients ui from equation 2.3
    u1 = (xi[0] * yi[1] * yi[2] + yi[0] * xi[1] * xi[2]) / (x - y)
    u2 = (xi[1] * yi[0] * yi[2] + yi[1] * xi[0] * xi[2]) / (x - y)
    u3 = (xi[2] * yi[0] * yi[1] + yi[2] * xi[0] * xi[1]) / (x - y)
    if (b[0] == 0 or b[1] == 0 or b[2] == 0 or b[3] == 0 or d14 == 0 or d24 == 0 or d34 == 0 or u1 == 0
            or xi[0] == 0 or yi[0] == 0):
        raise Exception('Error occurred in function elliptical_integral_cubic_all_roots_real. '
                        'We have situation where on of values which appear in denominator is zero')
    # coefficients w22 from equation 2.4
    w22 = u1 * u1 - b[3] * d12 * d13 / d14
    # coefficients q22 and p22 from equation 2.5
    q22 = w22 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    p22 = q22 + b[3] * d24 * d34 / d14
    # now I compute values of all R-functions
    rc_function_value = rc(p22, q22)
    rf_function_value = rf(u3 * u3, u2 * u2, u1 * u1)
    rd_function_value = rd(u3 * u3, u2 * u2, u1 * u1)
    rj_function_value = rj(u3 * u3, u2 * u2, u1 * u1, w22)
    # below we compute 3 version of integral which will be used later
    if p[3] == 0:
        # final integral is equal to double the value of R-functions (RF), equation 2.12
        return 2 * rf_function_value
    else:
        # here we do not compute the value of ffr, we used the value which was given as an input
        i1c = 2 * ffr
        # coef i3c from equation 2.14, here we use other R-functions (RJ and RC)
        i3c = -1 * b[0] * d12 * d13 * rj_function_value / 3.0 / d14
        i3c = i3c + 2 * rc_function_value
        if p[3] == -2:
            # final integral, equation 2.49
            return (b[3] * i3c - b[0] * i1c) / d14
        elif p[3] == -4:
            # coefficients r_{i,j} from equation 2.1
            r12 = d12 / b[0] / b[1]
            r13 = d13 / b[0] / b[2]
            r24 = d24 / b[1] / b[3]
            r34 = d34 / b[2] / b[3]
            # coef i2c from equation 2.13, here we use R-function (RD)
            i2c = 2 * d12 * d13 * rd_function_value / 3.0
            i2c = i2c + 2 * xi[0] * yi[0] / u1
            # coef from equation 2.6
            a111m2 = compute_a_p1_pn(p, a, b, y, x)
            # coef from equation 2.59
            k2c = b[1] * b[2] * i2c - 2 * b[3] * a111m2
            # final integral, equation 2.62
            return b[3] * k2c / (2 * d14 * d24 * d34) + i1c * (1 - r12 * r13 / 2.0 / r24 / r34) * (b[0] / d14) ** 2


def elliptical_integral_cubic_one_real_and_two_complex_roots(p: list, a: list, b: list, fgh: list, ffr: float,
                                                             y: float, x: float) -> float:
    """
        Integral computed based on numerical method by B. C; Carlson; A Table of Elliptic Integrals of the Third Kind
        Mathematics of Computation, Volume 51, Number 183, Pages 267-280, July 1988
        And
        B. C; Carlson; A Table of Elliptic Integrals: One Quadratic Factor
        Mathematics of Computation, Volume 56, Number 193, Pages 267-280, January 1991
    """

    if x < y:
        raise Exception('Error occurred in function elliptical_integral_cubic_one_real_root_two_complex_roots. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficient d14 from equation 2.1
    d14 = a[0] * b[3] - a[3] * b[0]
    if d14 == 0.0:
        raise Exception('Error occurred in function test_integrals_cubic_case_uw. We have situation where d14 = 0.')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.1
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    if xi[0] == 0.0 or yi[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_cubic_case_uw. '
                        'We have situation where one or both xi[0], yi[0] = 0.')
    # coefficient beta1 from equation 2.2
    beta1 = fgh[1] * b[0] - 2 * fgh[2] * a[0]
    # coefficients c11, c14, c44 from equation 2.3
    c112 = compute_cij_value(a, b, fgh, 0, 0)
    c142 = compute_cij_value(a, b, fgh, 0, 3)
    c442 = compute_cij_value(a, b, fgh, 3, 3)
    # coefficients ksi and eta given from equation 2.4
    ksi = math.sqrt(fgh[0] + fgh[1] * x + fgh[2] * x * x)
    eta = math.sqrt(fgh[0] + fgh[1] * y + fgh[2] * y * y)
    # value of M2 from equation 3.1
    m2 = ((xi[0] + yi[0]) * math.sqrt((ksi + eta) ** 2 - fgh[2] * (x - y) ** 2) / (x - y)) ** 2
    # values of Lm2, Lp2 and Wp2 from equation 3.2
    lm2 = m2 - beta1 - math.sqrt(2 * fgh[2] * c112)
    lp2 = m2 - beta1 + math.sqrt(2 * fgh[2] * c112)
    # value of R-function RF
    rf_function_value = rf(m2, lm2, lp2)
    if p[3] == 0:
        return 4.0 * rf_function_value
    else:
        i1c = ffr
        # value of Wp2 from equation 3.2
        wp2 = m2 - b[0] * (c142 + math.sqrt(c112 * c442)) / d14
        # values of coefficients U2 and W2 from equation 3.3
        u2 = ((xi[0] * eta + yi[0] * ksi) / (x - y)) ** 2
        w2 = u2 - c112 * b[3] / 2.0 / d14
        # values of coefficients Q2 and P2 from equation 3.4
        q2 = w2 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
        p2 = q2 + c442 * b[3] / 2.0 / d14
        # values of other R-functions RC, RD, RJ
        rc_function_value_uw = rc(u2, w2)
        rc_function_value_qp = rc(p2, q2)
        rd_function_value = rd(m2, lm2, lp2)
        rj_function_value = rj(m2, lm2, lp2, wp2)
        # additional coefficient from equation 3.5
        rho = math.sqrt(2 * fgh[2] * c112) - beta1
        if p[3] == -2:
            # Ic3 computed from equation 3.10
            i3c = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c442)) * \
                  ((-4 * b[0] / d14) * (c142 + math.sqrt(c112 * c442)) * rj_function_value
                   - 6 * rf_function_value + 3 * rc_function_value_uw) + 2 * rc_function_value_qp
            # return the final integral
            # the equation for integral is taken from Carlson paper from 1989 equation 2.49
            # but we use I3c and other functions computed in the way of paper 1991
            # we do it that way because here we have double complex root as a solution of cubic equation instead
            # 3 real solutions
            return (b[3] * i3c - b[0] * i1c) / d14
        elif p[3] == -4:
            # coefficient A(-1, 1, 1, -2, 0) needed for k2c value
            am111m2 = compute_a_p1_pn_quadratics(p[1], a, b, fgh, y, x)
            # value of coefficient N2c from equation 3.11
            n2c = math.sqrt(8 * fgh[2] / 9.0 / c112) * (4 * rho * rd_function_value - 6 * rf_function_value
                                                        + 3.0 / math.sqrt(u2)) + 2.0 / (xi[0] * yi[0] * math.sqrt(u2))
            # value of coefficient K2c from equation 3.12
            k2c = c112 * n2c / 2.0 - 2 * d14 * am111m2
            # we compute some combination of coefficients rij using identities 2.19
            r24r34 = c442 / 2.0 / b[3] / b[3] / fgh[2]
            r12r13 = c112 / 2.0 / b[0] / b[0] / fgh[2]

            # return the final integral
            # the equation for integral is taken from Carlson paper from 1989 equation 2.49
            # but we use I3c and other functions computed in the way of paper 1991
            # we do it that way because here we have double complex root as a solution of cubic equation instead
            # 3 real solutions
            return b[3] * k2c / 2.0 / d14 / c442 / 2.0 + (b[0] / d14) ** 2 * (1.0 - r12r13 / r24r34) * i1c


def elliptical_integral_quartic_all_real_roots(p: list, a: list, b: list, ffr: float, y: float, x: float) -> float:
    """
        Integral computed based on numerical method by B. C; Carlson; A Table of Elliptic Integrals of the Third Kind
        Mathematics of Computation, Volume 51, Number 183, Pages 267-280, July 1988
    """

    if x < y:
        raise Exception('Error occurred in function elliptical_integral_quartic_all_roots_real. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficients d_{i,j} from equation 2.1
    d12 = a[0] * b[1] - a[1] * b[0]
    d13 = a[0] * b[2] - a[2] * b[0]
    d14 = a[0] * b[3] - a[3] * b[0]
    d15 = a[0] * b[4] - a[4] * b[0]
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.2
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    # coefficients ui from equation 2.3
    u12 = (xi[0] * xi[1] * yi[2] * yi[3] + yi[0] * yi[1] * xi[2] * xi[3]) / (x - y)
    u13 = (xi[0] * xi[2] * yi[1] * yi[3] + yi[0] * yi[2] * xi[1] * xi[3]) / (x - y)
    u14 = (xi[0] * xi[3] * yi[1] * yi[2] + yi[0] * yi[3] * xi[1] * xi[2]) / (x - y)
    if (b[0] == 0 or b[1] == 0 or b[2] == 0 or b[4] == 0 or d15 == 0 or u14 == 0 or xi[0] == 0 or
            yi[0] == 0 or xi[3] == 0 or yi[3] == 0):
        raise Exception('Error occurred in function elliptical_integral_quartic_all_roots_real. '
                        'We have situation where on of values which appear in denominator is zero')
    if a[4] == 1.0 and b[4] == 0.0:
        # coefficients w122, q122 and p122 from equation 2.10
        w122 = u12 * u12 - b[1] * d13 * d14 / b[0]
        q122 = w122 / (xi[0] * yi[0]) ** 2
        p122 = q122 + b[1] * b[2] * b[3] / b[0]
        rc_function_value = rc(p122, q122)
        rf_function_value = rf(u12 * u12, u13 * u13, u14 * u14)
        # rd_function_value = rd(u12 * u12, u13 * u13, u14 * u14)
        rj_function_value = rj(u12 * u12, u13 * u13, u14 * u14, w122)
        i1 = 2.0 * rf_function_value
        # i2 = 2.0 * d12 * d13 * rd_function_value / 3.0 + 2.0 * xi[0] * yi[0] / xi[3] / yi[3] / u14
        i3p = -2.0 * d12 * d13 * d14 * rj_function_value / 3.0 / b[0] + 2.0 * rc_function_value
        if p[4] == 0:
            # final integral is equal to double the value of R-functions (RF), equation 2.13
            return i1
        elif p[4] == -2:
            # here i1 is not equal to the value of rf_function, but it is equal to the twice the value of ffr
            i1 = 2 * ffr
            # final integral, equation 2.35
            return (b[4] * i3p - b[0] * i1) / d15
    else:
        # additional coefficients d_{i,j} from equation 2.1
        d25 = a[1] * b[4] - a[4] * b[1]
        d35 = a[2] * b[4] - a[4] * b[2]
        d45 = a[3] * b[4] - a[4] * b[3]
        # coefficient w22 from equation 2.4
        w22 = u12 * u12 - d13 * d14 * d25 / d15
        # coefficients q22 and p22 from equation 2.5
        q22 = w22 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p22 = q22 + d25 * d35 * d45 / d15
        rc_function_value = rc(p22, q22)
        rf_function_value = rf(u12 * u12, u13 * u13, u14 * u14)
        rd_function_value = rd(u12 * u12, u13 * u13, u14 * u14)
        rj_function_value = rj(u12 * u12, u13 * u13, u14 * u14, w22)
        i1 = 2.0 * rf_function_value
        i2 = 2.0 * d12 * d13 * rd_function_value / 3.0 + 2.0 * xi[0] * yi[0] / xi[3] / yi[3] / u14
        i3 = 2.0 * d12 * d13 * d14 * rj_function_value / 3.0 / d15 + 2.0 * rc_function_value
        if p[4] == 0:
            # final integral is equal to double the value of R-functions (RF), equation 2.13
            return i1
        elif p[4] == -2:
            # here i1 is not equal to the value of rf_function, but it is equal to the twice the value of ffr
            i1 = 2 * ffr
            # final integral, equation 2.35
            return (b[4] * i3 - b[0] * i1) / d15
        elif p[4] == -4:
            # additional coefficients d_{i,j} from equation 2.1
            d24 = a[1] * b[3] - a[3] * b[1]
            d25 = a[1] * b[4] - a[4] * b[1]
            d34 = a[2] * b[3] - a[3] * b[2]
            d35 = a[2] * b[4] - a[4] * b[2]
            d45 = a[3] * b[4] - a[4] * b[3]
            # coefficients r_{i,j} from equation 2.1
            r12 = d12 / b[0] / b[1]
            r13 = d13 / b[0] / b[2]
            r25 = d25 / b[1] / b[4]
            r35 = d35 / b[2] / b[4]
            # coef from equation 2.6
            a111m1m2 = compute_a_p1_pn(p, a, b, y, x)
            # here i1 is not equal to the value of rf_function, but it is equal to the twice the value of ffr
            i1 = 2 * ffr
            # final integral, equation 2.49
            return b[4] * b[4] * d24 * d34 * i2 / (2 * d15 * d25 * d35 * d45) \
                   + b[0] * b[0] * i1 * (1 - r12 * r13 / (2 * r25 * r35)) - b[4] * b[4] * a111m1m2 / d15 / d25 / d35


def elliptical_integral_quartic_two_real_and_two_complex_roots(p: list, a: list, b: list, fgh: list, ffr: float,
                                                               y: float, x: float) -> float:
    """
        Integral computed based on numerical method by B. C; Carlson; A Table of Elliptic Integrals of the Third Kind
        Mathematics of Computation, Volume 51, Number 183, Pages 267-280, July 1988
        And
        B. C; Carlson; A Table of Elliptic Integrals: One Quadratic Factor
        Mathematics of Computation, Volume 56, Number 193, Pages 267-280, January 1991
    """
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_i123. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficient d14 from equation 2.1
    d14 = a[0] * b[3] - a[3] * b[0]
    d15 = a[0] * b[4] - a[4] * b[0]
    d45 = a[3] * b[4] - a[4] * b[3]
    if d15 == 0.0 or b[0] == 0.0 or fgh[2] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_i123. '
                        'We have situation where d15 = 0 or b[0] = 0 or h = 0.')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.1
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    if xi[0] == 0.0 or yi[0] == 0.0 or xi[3] == 0.0 or yi[3] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_i123. '
                        'We have situation where xi[0] = 0 or yi[0] = 0 or xi[3] = 0 or yi[3] = 0.')
    # coefficients c11, c14, c15, c44, c55 from equation 2.3
    c112 = compute_cij_value(a, b, fgh, 0, 0)
    c142 = compute_cij_value(a, b, fgh, 0, 3)
    c152 = compute_cij_value(a, b, fgh, 0, 4)
    c442 = compute_cij_value(a, b, fgh, 3, 3)
    c552 = compute_cij_value(a, b, fgh, 4, 4)
    if c442 == 0 or c552 == 0:
        raise Exception('Error occurred in function test_integrals_quartic_case_i123. '
                        'We have situation where c44 = 0 or c55 = 0.')
    # coefficients ksi and eta given from equation 2.4
    ksi = math.sqrt(fgh[0] + fgh[1] * x + fgh[2] * x * x)
    eta = math.sqrt(fgh[0] + fgh[1] * y + fgh[2] * y * y)
    # value of M2 from equation 2.6
    m2 = ((xi[0] * yi[3] + yi[0] * xi[3]) * math.sqrt((ksi + eta) ** 2 - fgh[2] * (x - y) ** 2) / (x - y)) ** 2
    # values of Lm2, Lp2 and Wp2 from equation 2.7
    lm2 = m2 + c142 - math.sqrt(c112 * c442)
    lp2 = m2 + c142 + math.sqrt(c112 * c442)
    wp2 = m2 + d14 * (c152 + math.sqrt(c112 * c552)) / d15
    # value of U2 from equation 2.8
    u2 = ((xi[0] * xi[3] * eta + yi[0] * yi[3] * ksi) / (x - y)) ** 2
    if u2 == 0:
        raise Exception('Error occurred in function test_integrals_quartic_case_i123. '
                        'We have situation where u2 = 0.')
    if a[4] == 1 and b[4] == 0:
        # value of W12 from equation 2.10
        w12 = u2 - c112 * b[3] / 2.0 / b[0]
        # value of rho from equation 2.11
        rho = d14 * (fgh[1] * b[0] - 2 * fgh[2] * a[0] - math.sqrt(2 * fgh[2] * c112)) / b[0]
        m2rho = m2 + rho
        # values of Q12 and P12 from equation 2.10
        q12 = w12 / xi[0] / xi[0] / yi[0] / yi[0]
        p12 = q12 + fgh[2] * b[3] / b[0]
        # R-functions values for different parameters
        rc_function_value_uw = rc(u2, w12)
        rc_function_value_qp = rc(p12, q12)
        rf_function_value = rf(m2, lm2, lp2)
        rd_function_value = rd(m2, lm2, lp2)
        rj_function_value = rj(m2, lm2, lp2, m2rho)
        if p[4] == 0:
            return 4 * rf_function_value
        elif p[4] == -2:
            i1 = ffr
            i3c = math.sqrt(2 * c112 / 9.0 / fgh[2]) * (4 * rho * rj_function_value - 6 * rf_function_value +
                                                        3 * rc_function_value_uw) + 2 * rc_function_value_qp
            # return the final integral
            # the equation for integral is taken from Carlson paper from 1989 equation 2.35
            # but we use I3c and other functions computed in the way of paper 1991
            # we do it that way because here we have double complex root and double real roots as a solution
            # of quartic equation instead 4 real solutions
            return -b[0] * i1 / d15
    else:
        # value of W2 from equation 2.8
        w2 = u2 - c112 * d45 / 2.0 / d15
        # values of Q2 and P2 from equation 2.9
        q2 = w2 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p2 = q2 + c552 * d45 / 2.0 / d15
        # R-functions values for different parameters
        rc_function_value_uw = rc(u2, w2)
        rc_function_value_qp = rc(p2, q2)
        rf_function_value = rf(m2, lm2, lp2)
        rd_function_value = rd(m2, lm2, lp2)
        rj_function_value = rj(m2, lm2, lp2, wp2)
        i1 = 4 * rf_function_value
        i2 = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c442)) * (4 * (c142 + math.sqrt(c112 * c442)) * rd_function_value
                                                              - 6 * rf_function_value + 3.0 / math.sqrt(u2)) \
             + 2 * xi[0] * yi[0] / xi[3] / yi[3] / math.sqrt(u2)
        i3 = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c552)) * (4 * d14 / d15 * (c152 + math.sqrt(c112 * c552))
                                                              * rj_function_value - 6 * rf_function_value
                                                              + 3 * rc_function_value_uw) + 2 * rc_function_value_qp
        if p[4] == 0:
            return 4 * rf_function_value
        elif p[4] == -2:
            i1 = ffr
            i3 = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c552)) * (4 * d14 / d15 * (c152 + math.sqrt(c112 * c552)) *
                                                                  rj_function_value - 6 * rf_function_value +
                                                                  3 * rc_function_value_uw) + 2 * rc_function_value_qp
            # return the final integral
            # the equation for integral is taken from Carlson paper from 1988 equation 2.35
            # but we use I3c and other functions computed in the way of paper 1991
            # we do it that way because here we have double complex root and double real roots as a solution
            # of quartic equation instead 4 real solutions
            return (b[4] * i3 - b[0] * i1) / d15
        elif p[4] == -4:
            i1 = ffr
            i2 = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c442)) * (4 * (c142 + math.sqrt(c112 * c442)) *
                                                                  rd_function_value - 6 * rf_function_value +
                                                                  3.0 / math.sqrt(u2)) + \
                 2 * xi[0] * yi[0] / xi[3] / yi[3] / math.sqrt(u2)
            # return the final integral
            # the equation for integral is taken from Carlson paper from 1988 equation 2.49
            # but we use I3c and other functions computed in the way of paper 1991
            # we do it that way because here we have double complex root and double real roots as a solution
            # of quartic equation instead 4 real solutions
            # here i use cii^2 instead of d2id3i as described in equation 2.19
            return b[4] * b[4] * c442 * i2 / 2.0 / d15 / d45 / c552 + b[0] * b[0] * i1 * \
                   (1 - c112 * b[4] * b[4] / 2.0 / c552 / b[0] / b[0]) \
                   - 2 * b[4] * b[4] / d15 / c552 * compute_a_p1_pn_quadratics(p, a, b, fgh, y, x)


def elliptical_integral_quartic_all_complex_roots(p: list, a: list, b: list, fgh1: list, fgh2: list, ffr: float,
                                                  y: float, x: float) -> float:
    """
        Integral computed based on numerical method by B. C; Carlson; A Table of Elliptic Integrals: Two Quadratic Factors
        Mathematics of Computation, Volume 59, Number 199, Pages 165-180, July 1992
    """
    if x < y:
        raise Exception('Error occurred in function elliptical_integral_quartic_all_complex_roots. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    fi = [fgh1[0], fgh2[0]]
    gi = [fgh1[1], fgh2[1]]
    hi = [fgh1[2], fgh2[2]]
    ksi = []
    eta = []
    theta = []
    # coefficient ksi and eta from equation 2.1
    for i in range(len(fi)):
        ksi.append(math.sqrt(fi[i] + gi[i] * x + hi[i] * x * x))
        eta.append(math.sqrt(fi[i] + gi[i] * y + hi[i] * y * y))
        theta.append(ksi[i] * ksi[i] + eta[i] * eta[i] - hi[i] * (x - y) ** 2)
    # coefficients from equation 2.2
    ksi1p = (gi[0] + 2 * hi[0] * x) / 2.0 / ksi[0]
    eta1p = (gi[0] + 2 * hi[0] * y) / 2.0 / eta[0]
    dzeta = []
    # coefficient dzeta from equation 2.5
    for i in range(len(fi)):
        dzeta.append(math.sqrt((ksi[i] + eta[i]) ** 2 - hi[i] * (x - y) ** 2))
    # coefficient M from equation 2.6
    u = (ksi[0] * eta[1] + eta[0] * ksi[1]) / (x - y)
    m = dzeta[0] * dzeta[1] / (x - y)
    m2 = m * m
    # coefficients delta_ij and delta from equation 2.7
    d112 = compute_delta_ij(fi, gi, hi, 0, 0)
    d222 = compute_delta_ij(fi, gi, hi, 1, 1)
    d122 = compute_delta_ij(fi, gi, hi, 0, 1)
    delta = math.sqrt(d122 * d122 - d112 * d222)
    # coefficients delta_plus/minus and Lm2 and Lp2 from equation 2.8
    deltap = d122 + delta
    deltam = d122 - delta
    lm2 = m2 + deltam
    lp2 = m2 + deltap
    # values of RF and RD functions
    rf_function_value = rf(m2, lm2, lp2)
    rd_function_value = rd(m2, lm2, lp2)
    if p[4] == 0:
        # return the final integral from equation 2.36
        return 4 * rf_function_value
    if u == 0.0 or ksi[0] == 0.0 or eta[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_m2lm2lp2. '
                        'We have situation where ksi[0] = 0 or eta[0] = 0 or u = 0.0.')
    # value of coefficient G from equation 2.9
    gie = 2 * delta * deltap * rd_function_value / 3.0 + delta / 2.0 / u + \
          (d122 * theta[0] - d112 * theta[1]) / 4.0 / ksi[0] / eta[0] / u
    # coefficients alpha, beta and gamma from equations 2.11 and 2.12
    alfa15, beta15 = compute_alfa_beta_i5(fi, gi, hi, a[4], b[4], 0)
    alfa25, beta25 = compute_alfa_beta_i5(fi, gi, hi, a[4], b[4], 1)
    gamma1 = compute_gamma_i5(fi, gi, hi, a[4], b[4], 0)
    gamma2 = compute_gamma_i5(fi, gi, hi, a[4], b[4], 1)
    if a[4] == 1.0 and b[4] == 0.0:
        if hi[0] == 0.0:
            raise Exception('Error occurred in function test_integrals_quartic_case_m2lm2lp2. '
                            'We have situation where hi[0] = 0.')
        # coefficients capital lambda and capital sigma from equation 2.23
        blambda = d112 * hi[1] / hi[0]
        bsig2 = m2 + blambda
        # coefficient psi form equation 2.24
        psi = gi[0] * hi[1] - gi[1] * hi[0]
        # value of capital x from equation 2.25
        bix = -1.0 * (ksi1p * ksi[1] + eta1p * eta[1]) / (x - y)
        # value of capital s from equation 2.18
        bes = (m2 + d122) / 2.0 - u * u
        # value of coefficient mu and values of capital t and capital V squared from equation 2.19
        mu = hi[0] / ksi[0] / eta[0]
        bet = mu * bes + 2 * hi[0] * hi[1]
        v2 = mu * mu * (bes * bes + blambda * u * u)
        # values of b2 and a2 from equation 2.20 and 2.21 respectively
        b2 = (bes * bes / u / u + blambda) * bsig2 * bsig2
        a2 = b2 + blambda * (deltap - blambda) * (blambda - deltam)
        # values of RC and RJ functions for different arguments
        rca2b2 = rc(a2, b2)
        rct2v2 = rc(bet * bet, v2)
        rjm2lm2lp2sig2 = rj(m2, lm2, lp2, bsig2)
        # values of capital h from equation 2.22
        beh = d112 * psi * (rjm2lm2lp2sig2 / 3.0 + rca2b2 / 2.0) / hi[0] / hi[0] - bix * rct2v2
        if p[4] == -2:
            # return the final integral from equation 2.39
            return -2 * (b[4] * beh + beta15 * ffr / gamma1)
        elif p[4] == -4:
            # coefficient omega from equation 2.10
            omega = gie - deltap * ffr + ksi1p * ksi[0] - eta1p * eta[0]
            # return the final integral from equation 2.41
            return b[4] * (beta15 / gamma1 + beta25 / gamma2) * beh + beta15 * beta15 * ffr / gamma1 / gamma1 + \
                   b[4] * b[4] * (omega - b[4] * compute_a_p1_pn_double_quadratic(p, a, b, fgh1, fgh2, y, x)) \
                   / gamma1 / gamma2
    else:
        # coefficients capital lambda and capital sigma from equation 2.13
        blambda = d112 * gamma2 / gamma1
        bsig2 = m2 + blambda
        # coefficient psi form equation 2.14
        psi = (alfa15 * beta25 - alfa25 * beta15) / 2.0
        # values of ksi5 and eta5 from equation 2.15
        ksi5 = a[4] + b[4] * x
        eta5 = a[4] + b[4] * y
        # value of capital x from equation 2.17
        bix = (ksi5 * (alfa15 + beta15 * y) * eta[1] / eta[0] + eta5 * (alfa15 + beta15 * x) * ksi[1] / ksi[0]) \
              / 2.0 / (x - y)
        # value of capital s from equation 2.18
        bes = (m2 + d122) / 2.0 - u * u
        # value of coefficient mu and values of capital t and capital V squared from equation 2.19
        mu = gamma1 * ksi5 * eta5 / ksi[0] / eta[0]
        bet = mu * bes + 2 * gamma1 * gamma2
        v2 = mu * mu * (bes * bes + blambda * u * u)
        # values of b2 and a2 from equation 2.20 and 2.21 respectively
        b2 = (bes * bes / u / u + blambda) * bsig2 * bsig2
        a2 = b2 + blambda * (deltap - blambda) * (blambda - deltam)
        # values of RC and RJ functions for different arguments
        rca2b2 = rc(a2, b2)
        rct2v2 = rc(bet * bet, v2)
        rjm2lm2lp2sig2 = rj(m2, lm2, lp2, bsig2)
        # values of capital h from equation 2.22
        beh = d112 * psi * (rjm2lm2lp2sig2 / 3.0 + rca2b2 / 2.0) / gamma1 / gamma1 - bix * rct2v2
        if p[4] == -2:
            # return the final integral from equation 2.39
            return -2 * (b[4] * beh + beta15 * ffr / gamma1)
        elif p[4] == -4:
            # coefficient omega from equation 2.10
            omega = gie - deltap * ffr + ksi1p * ksi[0] - eta1p * eta[0]
            # return the final integral from equation 2.41
            return b[4] * (beta15 / gamma1 + beta25 / gamma2) * beh + beta15 * beta15 * ffr / gamma1 / gamma1 + \
                   b[4] * b[4] * (omega - b[4] * compute_a_p1_pn_double_quadratic(p, a, b, fgh1, fgh2, y, x)) \
                   / gamma1 / gamma2
