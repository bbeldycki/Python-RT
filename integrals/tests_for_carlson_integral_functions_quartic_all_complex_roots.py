from integrals import integrals
import math
import time


def carlson_intermediate_values_quartic_case() -> list:
    # Function which return a list of some intermediate values for given input
    # f1 = 2.7, g1 = -1.8, h1 = 0.9, f2= 2.0, g2 = 2.4, h2 = 0.8, a5 = 1.1 b5 = -0.4, x = 2.0, y = -3.0
    m2 = 0.36362947     # index 0
    lm2 = 0.53423014    # index 1
    lp2 = 24.673029     # index 2
    sig2 = 21.199185    # index 3
    sig02 = 6.1236295   # index 4
    a2 = 11237.193      # index 5
    b2 = 9741.4746      # index 6
    a02 = 844.71933     # index 7
    b02 = 247.52253     # index 8
    t2 = 10.288757      # index 9
    v2 = 1.1362990      # index 10
    t02 = 1.1328716     # index 11
    v02 = 1.1077327     # index 12
    ix = -1.1571677     # index 13
    ix0 = -0.093427949      # index 14
    rca2b2 = 0.0098889795  # index 15
    rca02b02 = 0.050085175  # index 16
    rct2v2 = 0.58372845     # index 17
    rct02v02 = 0.94657139   # index 18
    rfm2lm2lp2 = 0.54784092       # index 19
    rdm2lm2lp2 = 0.042910488      # index 20
    rjm2lm2lp2sig2 = 0.048599080  # index 21
    rjm2lm2lp2sig02 = 0.12739513  # index 22
    gie = 10.495586         # index 23
    ha = 0.049905556        # index 24
    ha0 = -1.8557835        # index 25
    am111m1 = 1.5731367     # index 26
    a1111 = -0.49594737     # index 27
    am111m1m2 = 6.2622360   # index 28
    a1111m2 = 14.845682     # index 29
    result_list = [m2, lm2, lp2, sig2, sig02, a2, b2, a02, b02, t2, v2, t02, v02, ix, ix0, rca2b2, rca02b02, rct2v2,
                   rct02v02, rfm2lm2lp2, rdm2lm2lp2, rjm2lm2lp2sig2, rjm2lm2lp2sig02, gie, ha, ha0, am111m1, a1111,
                   am111m1m2, a1111m2]
    return result_list


def quartic_cases_test_data() -> tuple:
    a_list = [0.0, 0.0, 0.0, 0.0, 1.1]
    b_list = [0.0, 0.0, 0.0, 0.0, -0.4]
    # a_list = [0.0, 0.0, 0.0, 0.0, 1.0]
    # b_list = [0.0, 0.0, 0.0, 0.0, 0.0]
    fgh_1_list = [2.7, -1.8, 0.9]
    fgh_2_list = [2.0, 2.4, 0.8]
    y_val = -3.0
    x_val = 2.0
    p_list = [[-1, -1, -1, -1, 0], [-1, 1, 1, -1, 0], [1, 1, 1, 1, 0], [-1, 1, 1, -1, -2], [1, 1, 1, 1, -2]]
    # p_list is a list of lists
    # zero could be skipped but I will leave it to make sure every list has the same length
    # by default we will use p_list[0] = [ -1, -1, -1, 0, 0]
    # [-1, 1, 1, -1, 0]     index 1
    # [1, 1, 1, 1, 0]       index 2
    # [-1, 1, 1, -1, -2]    index 3
    # [1, 1, 1, 1, -2]      index 4
    return p_list, a_list, b_list, fgh_1_list, fgh_2_list, y_val, x_val


def comparison_result_lists(list1: list, list2: list, eps: float) -> bool:
    # Function to compare my computed values with values from Carlson computations
    # Remember to put my computed values in list1 and Carlson values in list2
    if len(list1) != len(list2):
        raise Exception('Error: Len of list1 is not equal to Len of list2.')
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > eps:
            raise Exception('Error occurred during comparison of ours computed results with values from carlson '
                            'article. Reason: Absolute value of difference is larger than one part in a million. '
                            f'Error occurred for i value {i}, with my result = {list1[i]} and carlson = {list2[i]}')
    return True


def test_integrals_quartic_case_a_p1_pn(p: list, a: list, b: list, fgh1: list, fgh2: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_a_p1_pn. '
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


def compute_alfa_beta_i5(f: list, g: list, h: list, a5: float, b5: float, k: int) -> tuple:
    # Function which computes coefficients alfa_i5 and beta_i5
    return 2 * f[k] * b5 - g[k] * a5, g[k] * b5 - 2 * h[k] * a5


def compute_delta_ij(f: list, g: list, h: list, k: int, l: int) -> float:
    # Function which computes coefficients delta_ij
    return 2 * f[k] * h[l] + 2 * f[l] * h[k] - g[k] * g[l]


def compute_gamma_i5(f: list, g: list, h: list, a5: float, b5: float, k: int) -> float:
    # Function which computes coefficients gamma_i5
    gama = f[k] * b5 * b5 - g[k] * a5 * b5 + h[k] * a5 * a5
    if gama > 0.0:
        return gama
    else:
        raise Exception('Error occurred during comparison of gamma_i5 in function compute_gamma_i5. '
                        'Resulting value was negative or zero,but should be greater than zero ')


def test_integrals_quartic_case_m2lm2lp2(fgh1: list, fgh2: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_m2lm2lp2. '
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
    rf_function_value = integrals.rf(m2, lm2, lp2)
    rd_function_value = integrals.rd(m2, lm2, lp2)
    if u == 0.0 or ksi[0] == 0.0 or eta[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_m2lm2lp2. '
                        'We have situation where ksi[0] = 0 or eta[0] = 0 or u = 0.0.')
    # value of coefficient G from equation 2.9
    gie = 2 * delta * deltap * rd_function_value / 3.0 + delta / 2.0 / u + \
          (d122 * theta[0] - d112 * theta[1]) / 4.0 / ksi[0] / eta[0] / u
    return [m2, lm2, lp2, rf_function_value, rd_function_value, gie]


def test_integrals_quartic_case_pne0(a: list, b: list, fgh1: list, fgh2: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_pne0. '
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
    if u == 0.0 or ksi[0] == 0.0 or eta[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_m2lm2lp2. '
                        'We have situation where ksi[0] = 0 or eta[0] = 0 or u = 0.0.')
    if a[4] == 1.0 and b[4] == 0.0:
        # coefficients from equation 2.2
        ksi1p = (gi[0] + 2 * hi[0] * x) / 2.0 / ksi[0]
        eta1p = (gi[0] + 2 * hi[0] * y) / 2.0 / eta[0]
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
        rca2b2 = integrals.rc(a2, b2)
        rct2v2 = integrals.rc(bet * bet, v2)
        rjm2lm2lp2sig2 = integrals.rj(m2, lm2, lp2, bsig2)
        # values of capital h from equation 2.22
        beh = d112 * psi * (rjm2lm2lp2sig2 / 3.0 + rca2b2 / 2.0) / hi[0] / hi[0] - bix * rct2v2
        return [bsig2, a2, b2, bet * bet, v2, bix, rca2b2, rct2v2, rjm2lm2lp2sig2, beh]
    else:
        # coefficients alpha, beta and gamma from equations 2.11 and 2.12
        alfa15, beta15 = compute_alfa_beta_i5(fi, gi, hi, a[4], b[4], 0)
        alfa25, beta25 = compute_alfa_beta_i5(fi, gi, hi, a[4], b[4], 1)
        gamma1 = compute_gamma_i5(fi, gi, hi, a[4], b[4], 0)
        gamma2 = compute_gamma_i5(fi, gi, hi, a[4], b[4], 1)
        if gamma1 == 0.0:
            raise Exception('Error occurred in function test_integrals_quartic_case_m2lm2lp2. '
                            'We have situation where gamma1 = 0.')
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
        rca2b2 = integrals.rc(a2, b2)
        rct2v2 = integrals.rc(bet * bet, v2)
        rjm2lm2lp2sig2 = integrals.rj(m2, lm2, lp2, bsig2)
        # values of capital h from equation 2.22
        beh = d112 * psi * (rjm2lm2lp2sig2 / 3.0 + rca2b2 / 2.0) / gamma1 / gamma1 - bix * rct2v2
        return [bsig2, a2, b2, bet * bet, v2, bix, rca2b2, rct2v2, rjm2lm2lp2sig2, beh]


if __name__ == "__main__":
    start = time.time()
    p_test, a_test, b_test, fgh1_test, fgh2_test, y_test, x_test = quartic_cases_test_data()
    data_to_compare_with = carlson_intermediate_values_quartic_case()
    results_m2lm2lp2 = test_integrals_quartic_case_m2lm2lp2(fgh1_test, fgh2_test, y_test, x_test)
    eps = 1e-6
    if comparison_result_lists(results_m2lm2lp2, [data_to_compare_with[0], data_to_compare_with[1],
                                                  data_to_compare_with[2], data_to_compare_with[19],
                                                  data_to_compare_with[20], data_to_compare_with[23]], eps=eps):
        print(f'Correct values of: M2, Lm2, Lp2, G; RF and RD functions with eps = {eps}')
    results_coefandrfun = test_integrals_quartic_case_pne0(a_test, b_test, fgh1_test, fgh2_test, y_test, x_test)
    eps = 1e-3
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result_lists(results_coefandrfun, [data_to_compare_with[4], data_to_compare_with[7],
                                                         data_to_compare_with[8], data_to_compare_with[11],
                                                         data_to_compare_with[12], data_to_compare_with[14],
                                                         data_to_compare_with[16], data_to_compare_with[18],
                                                         data_to_compare_with[22], data_to_compare_with[25]], eps=eps):
            print(f'Correct values of: Sig2, a2, b2, T2, V2, X, H; RC and RJ functions with eps = {eps}')
    else:
        if comparison_result_lists(results_coefandrfun, [data_to_compare_with[3], data_to_compare_with[5],
                                                         data_to_compare_with[6], data_to_compare_with[9],
                                                         data_to_compare_with[10], data_to_compare_with[13],
                                                         data_to_compare_with[15], data_to_compare_with[17],
                                                         data_to_compare_with[21], data_to_compare_with[24]], eps=eps):
            print(f'Correct values of: Sig2, a2, b2, T2, V2, X, H; RC and RJ functions with eps = {eps}')
    result_a_p1_pn = []
    list_of_data_to_compare_with = []
    for i in range(4):
        result_a_p1_pn.append(test_integrals_quartic_case_a_p1_pn(p_test[i+1], a_test, b_test, fgh1_test,
                                                                  fgh2_test, y_test, x_test))
        list_of_data_to_compare_with.append(data_to_compare_with[i + 26])
    if comparison_result_lists(result_a_p1_pn, list_of_data_to_compare_with, 1e-6):
        print(f'Correct values of A for different p[i] with eps = {1e-6}')
    koniec = time.time()
    time = (koniec - start) * 1000
    print(f'Tests took: {time} ms')