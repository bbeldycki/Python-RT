from integrals import integrals
import math
import time


def carlson_intermediate_values_quartic_case() -> list:
    # Function which return a list of some intermediate values for given input
    # a1 = 0,3. a4 = 0.9, a5 = 0.4, b1 = 0.2, b4 = -0.3, b5 = 0.5, f = 0.4, g = -0.2, h = 0.1, x = 2.0, y = 0.5
    m2 = 0.62249271     # index 0
    lm2 = 0.54993185    # index 1
    lp2 = 0.74305357    # index 2
    wp2 = -0.54216139   # index 3
    u2 = 0.16410988     # index 4
    w2 = -0.13717583    # index 5
    m2prho = 0.92172730     # index 6
    w12 = 0.21960988    # index 7
    rcu2w2 = 1.7237432  # index 8
    rcp2q2 = 0.98880184     # index 9
    rcu2w12 = 2.2358652     # index 10
    rcp12q12 = 1.16864877   # index 11
    rfm2lm2lp2 = 1.2543726  # index 12
    rdm2lm2lp2 = 1.7960842  # index 13
    rjm2lm2lp2wp2 = -0.99822609     # index 14
    rjm2lm2lp2m2prho = 1.5689637    # index 15
    i1 = 5.0174903      # index 16
    i2 = 5.8882786      # index 17
    i3 = 2.7228427      # index 18
    i3p = 2.7668674     # index 19
    a111m1 = 0.54975858     # index 20
    a111m1m2 = 0.04955294   # index 21
    am111m1 = 0.33929812    # index 22
    a111m3 = 2.6651950      # index 23
    result_list = [m2, lm2, lp2, wp2, u2, w2, m2prho, w12, rcu2w2, rcp2q2, rcu2w12, rcp12q12, rfm2lm2lp2, rdm2lm2lp2,
                   rjm2lm2lp2wp2, rjm2lm2lp2m2prho, i1, i2, i3, i3p, a111m1, a111m1m2, am111m1, a111m3]
    return result_list


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


def compute_cij_value(a: list, b: list, fgh: list, k: int, k1: int) -> float:
    # Function which computes coefficients cij
    return 2 * b[k] * b[k1] * fgh[0] - fgh[1] * (a[k] * b[k1] + a[k1] * b[k]) + 2 * fgh[2] * a[k] * a[k1]


def quartic_cases_test_data() -> tuple:
    a_list = [0.3, 0.0, 0.0, 0.9, 0.4]
    b_list = [0.2, 0.0, 0.0, -0.3, 0.5]
    # a_list = [0.3, 0.0, 0.0, 0.9, 1.0]
    # b_list = [0.2, 0.0, 0.0, -0.3, 0.0]
    fgh_list = [0.4, -0.2, 0.1]
    y_val = 0.5
    x_val = 2.0
    p_list = [[-1, -1, -1, 0, 0], [1, 1, 1, -1, 0], [1, 1, 1, -1, -2], [-1, 1, 1, -1, 0], [1, 1, 1, -3, 0]]
    # p_list is a list of lists
    # zero could be skipped but I will leave it to make sure every list has the same length
    # by default we will use p_list[0] = [ -1, -1, -1, 0, 0]
    # [1, 1, 1, -1, 0]      index 1
    # [1, 1, 1, -1, -2]     index 2
    # [-1, 1, 1, -1, 0]     index 3
    # [1, 1, 1, -3, 0]      index 4
    return p_list, a_list, b_list, fgh_list, y_val, x_val


def test_integrals_quartic_case_a_p1_pn(p: list, a: list, b: list, fgh: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_a_p1_pn. '
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


def test_integrals_quartic_case_i123(a: list, b: list, fgh: list, y: float, x: float) -> list:
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
        rc_function_value_uw = integrals.rc(u2, w12)
        rc_function_value_qp = integrals.rc(p12, q12)
        rf_function_value = integrals.rf(m2, lm2, lp2)
        rd_function_value = integrals.rd(m2, lm2, lp2)
        rj_function_value = integrals.rj(m2, lm2, lp2, m2rho)
        i3c = math.sqrt(2 * c112 / 9.0 / fgh[2]) * (4 * rho * rj_function_value - 6 * rf_function_value +
                                                   3 * rc_function_value_uw) + 2 * rc_function_value_qp
        return [i3c]
    else:
        # value of W2 from equation 2.8
        w2 = u2 - c112 * d45 / 2.0 / d15
        # values of Q2 and P2 from equation 2.9
        q2 = w2 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p2 = q2 + c552 * d45 / 2.0 / d15
        # R-functions values for different parameters
        rc_function_value_uw = integrals.rc(u2, w2)
        rc_function_value_qp = integrals.rc(p2, q2)
        rf_function_value = integrals.rf(m2, lm2, lp2)
        rd_function_value = integrals.rd(m2, lm2, lp2)
        rj_function_value = integrals.rj(m2, lm2, lp2, wp2)
        i1 = 4 * rf_function_value
        i2 = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c442)) * (4 * (c142 + math.sqrt(c112 * c442)) * rd_function_value
                                                              - 6 * rf_function_value + 3.0 / math.sqrt(u2)) \
             + 2 * xi[0] * yi[0] / xi[3] / yi[3] / math.sqrt(u2)
        i3 = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c552)) * (4 * d14 / d15 * (c152 + math.sqrt(c112 * c552))
                                                              * rj_function_value - 6 * rf_function_value
                                                              + 3 * rc_function_value_uw) + 2 * rc_function_value_qp
        return [i1, i2, i3]


def test_integrals_quartic_case_lwu(a: list, b: list, fgh: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_lwu. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficient d14 from equation 2.1
    d14 = a[0] * b[3] - a[3] * b[0]
    d15 = a[0] * b[4] - a[4] * b[0]
    d45 = a[3] * b[4] - a[4] * b[3]
    if d15 == 0.0 or b[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_lwu. '
                        'We have situation where d15 = 0 or b[0] = 0.')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.1
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    # coefficients c11, c14, c15, c44, c55 from equation 2.3
    c112 = compute_cij_value(a, b, fgh, 0, 0)
    c142 = compute_cij_value(a, b, fgh, 0, 3)
    c152 = compute_cij_value(a, b, fgh, 0, 4)
    c442 = compute_cij_value(a, b, fgh, 3, 3)
    c552 = compute_cij_value(a, b, fgh, 4, 4)
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
    if a[4] == 1 and b[4] == 0:
        # value of W12 from equation 2.10
        w12 = u2 - c112 * b[3] / 2.0 / b[0]
        rho = d14 * (fgh[1] * b[0] - 2 * fgh[2] * a[0] - math.sqrt(2 * fgh[2] * c112)) / b[0]
        m2rho = m2 + rho
        return [m2, lm2, lp2, w12, m2rho]
    else:
        # value of W2 from equation 2.8
        w2 = u2 - c112 * d45 / 2.0 / d15
        return [m2, lm2, lp2, wp2, u2, w2]


def test_integrals_quartic_case_rcrfrdrj(a: list, b: list, fgh: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_case_rcrfrdrj. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficient d14 from equation 2.1
    d14 = a[0] * b[3] - a[3] * b[0]
    d15 = a[0] * b[4] - a[4] * b[0]
    d45 = a[3] * b[4] - a[4] * b[3]
    if d15 == 0.0 or b[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_rcrfrdrj. '
                        'We have situation where d15 = 0 or b[0] = 0.')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.1
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    if xi[0] == 0.0 or yi[0] == 0.0:
        raise Exception('Error occurred in function test_integrals_quartic_case_rcrfrdrj. '
                        'We have situation where xi[0] = 0 or yi[0] = 0.')
    # coefficients c11, c14, c15, c44, c55 from equation 2.3
    c112 = compute_cij_value(a, b, fgh, 0, 0)
    c142 = compute_cij_value(a, b, fgh, 0, 3)
    c152 = compute_cij_value(a, b, fgh, 0, 4)
    c442 = compute_cij_value(a, b, fgh, 3, 3)
    c552 = compute_cij_value(a, b, fgh, 4, 4)
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
        rc_function_value_uw = integrals.rc(u2, w12)
        rc_function_value_qp = integrals.rc(p12, q12)
        rf_function_value = integrals.rf(m2, lm2, lp2)
        rd_function_value = integrals.rd(m2, lm2, lp2)
        rj_function_value = integrals.rj(m2, lm2, lp2, m2rho)
        return [rc_function_value_uw, rc_function_value_qp, rf_function_value, rd_function_value, rj_function_value]
    else:
        # value of W2 from equation 2.8
        w2 = u2 - c112 * d45 / 2.0 / d15
        # values of Q2 and P2 from equation 2.9
        q2 = w2 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p2 = q2 + c552 * d45 / 2.0 / d15
        # R-functions values for different parameters
        rc_function_value_uw = integrals.rc(u2, w2)
        rc_function_value_qp = integrals.rc(p2, q2)
        rf_function_value = integrals.rf(m2, lm2, lp2)
        rd_function_value = integrals.rd(m2, lm2, lp2)
        rj_function_value = integrals.rj(m2, lm2, lp2, wp2)
        return [rc_function_value_uw, rc_function_value_qp, rf_function_value, rd_function_value, rj_function_value]


if __name__ == "__main__":
    start = time.time()
    p_test, a_test, b_test, fgh_test, y_test, x_test = quartic_cases_test_data()
    data_to_compare_with = carlson_intermediate_values_quartic_case()
    result_l2w2u2 = test_integrals_quartic_case_lwu(a_test, b_test, fgh_test, y_test, x_test)
    eps = 1e-6
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result_lists(result_l2w2u2, [data_to_compare_with[0],
                                                   data_to_compare_with[1], data_to_compare_with[2],
                                                   data_to_compare_with[7], data_to_compare_with[6]], eps=eps):
            print(f'Correct values of: M2, Lm2, Lp2, Wp2, W12 and M2+rho with eps = {eps}')
    else:
        if comparison_result_lists(result_l2w2u2, [data_to_compare_with[0],
                                                   data_to_compare_with[1], data_to_compare_with[2],
                                                   data_to_compare_with[3], data_to_compare_with[4],
                                                   data_to_compare_with[5]], eps=eps):
            print(f'Correct values of: M2, Lm2, Lp2, Wp2, U2 and W2 with eps = {eps}')
    result_rcrfrdrj = test_integrals_quartic_case_rcrfrdrj(a_test, b_test, fgh_test, y_test, x_test)
    eps = 1e-6
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result_lists(result_rcrfrdrj, [data_to_compare_with[10], data_to_compare_with[11],
                                                     data_to_compare_with[12], data_to_compare_with[13],
                                                     data_to_compare_with[15]], eps=eps):
            print(f'Correct values of: RC, RF, RD, RJ with eps = {eps}')
    else:
        if comparison_result_lists(result_rcrfrdrj, [data_to_compare_with[8], data_to_compare_with[9],
                                                     data_to_compare_with[12], data_to_compare_with[13],
                                                     data_to_compare_with[14]], eps=eps):
            print(f'Correct values of: RC, RF, RD, RJ with eps = {eps}')
    result_i13 = test_integrals_quartic_case_i123(a_test, b_test, fgh_test, y_test, x_test)
    eps = 1e-6
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result_lists(result_i13, [data_to_compare_with[19]], eps=eps):
            print(f'Correct value of I3c with eps = {eps}')
    else:
        if comparison_result_lists(result_i13, [data_to_compare_with[16], data_to_compare_with[17],
                                                data_to_compare_with[18]], eps=eps):
            print(f'Correct values of: I1, I2 and I3 with eps = {eps}')
    result_a_p1_pn = []
    list_of_data_to_compare_with = []
    for i in range(4):
        result_a_p1_pn.append(test_integrals_quartic_case_a_p1_pn(p_test[i+1], a_test, b_test, fgh_test,
                                                                  y_test, x_test))
        list_of_data_to_compare_with.append(data_to_compare_with[i + 20])
    print(result_a_p1_pn)
    print(list_of_data_to_compare_with)
    if comparison_result_lists(result_a_p1_pn, list_of_data_to_compare_with, 1e-6):
        print(f'Correct values of A for different p[i] with eps = {1e-6}')
    koniec = time.time()
    time = (koniec - start) * 1000
    print(f'Tests took: {time} ms')