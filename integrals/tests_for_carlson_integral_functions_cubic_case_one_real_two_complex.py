from integrals import integrals
import math
import time


def carlson_intermediate_values_cubic_case() -> list:
    # Function which return a list of some intermediate values for given input
    # a1 = 0,3. a4 = 0.9, a5 = 0.4, b1 = 0.2, b4 = -0.3, b5 = 0.5, f = 0.4, g = -0.2, h = 0.1, x = 2.0, y = 0.5
    m2 = 1.1713435      # index 0
    lm2 = 1.1496883     # index 1
    lp2 = 1.3929988     # index 2
    wp2 = 1.2606479     # index 3
    u2 = 0.34181141     # index 4
    w2 = 0.30070030     # index 5
    rcu2w2 = 1.7844272  # index 6
    rcp2q2 = 1.9470611  # index 7
    rfm2lm2lp2 = 0.89978529  # index 8
    rdm2lm2lp2 = 0.67751039  # index 9
    rjm2lm2lp2wp2 = 0.71986645      # index 10
    i1c = 3.5991412     # index 11
    i2c = 1.9453098     # index 12
    i3c = 4.0022901     # index 13
    j1c = 0.06573017    # index 14
    n2c = 6.8301223     # index 15
    k1c = 0.96438740    # index 16
    am1m1m1 = -0.88367862   # index 17
    am111 = -0.14545887     # index 18
    am111m2 = 1.3179127     # index 19
    a111 = 0.16859514   # index 20
    am311 = -1.1735711  # index 21
    a311 = 0.22618313   # index 22
    result_list = [m2, lm2, lp2, wp2, u2, w2, rcu2w2, rcp2q2, rfm2lm2lp2, rdm2lm2lp2, rjm2lm2lp2wp2, i1c,
                   i2c, i3c, j1c, n2c, k1c, am1m1m1, am111, am111m2, a111, am311, a311]
    return result_list


def comparison_result_floats(value1: float, value2: float, eps: float) -> bool:
    # Function to compare my computed values with values from Carlson computations
    if abs(value1 - value2) > eps:
        raise Exception('Error occurred during comparison of ours computed results with values from carlson '
                        f'article. Reason: Absolute value of difference is larger than {eps}.')
    return True


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
    return 2 * b[k] * b[k1] * fgh[0] - fgh[1] * (a[k] * b[k1] + a[k1] * b[k]) + 2 * fgh[2] * a[k] * a[k1]


def cubic_cases_test_data() -> tuple:
    a_list = [0.3, 0.0, 0.0, 0.9, 0.4]
    b_list = [0.2, 0.0, 0.0, -0.3, 0.5]
    # a_list = [0.3, 0.0, 0.0, 1.0, 0.9]
    # b_list = [0.2, 0.0, 0.0, 0.0, -0.3]
    fgh_list = [0.4, -0.2, 0.1]
    y_val = 0.5
    x_val = 2.0
    p_list = [[-1, -1, -1, 0, 0], [-1, 1, 1, 0, 0], [-1, 1, 1, -2, 0], [1, 1, 1, 0, 0], [-3, 1, 1, 0, 0],
              [3, 1, 1, 0, 0]]
    # p_list is a list of lists
    # zero could be skipped but I will leave it to make sure every list has the same length
    # by default we will use p_list[0] = [ -1, -1, -1, 0, 0]
    # [-1, 1, 1, 0, 0]      index 1
    # [-1, 1, 1, -2, 0]     index 2
    # [1, 1, 1, 0, 0]       index 3
    # [-3, 1, 1, 0, 0]      index 4
    # [3, 1, 1, 0, 0]       index 5
    return p_list, a_list, b_list, fgh_list, y_val, x_val


def test_integrals_cubic_case_a_p1_pn(p: list, a: list, b: list, fgh: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_a_p1_pn. '
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


def test_integrals_cubic_case_functions_rcrfrdrj(a: list, b: list, fgh: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_functions_rcrfrdrj. '
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
    wp2 = m2 - b[0] * (c142 + math.sqrt(c112 * c442)) / d14
    # values of coefficients U2 and W2 from equation 3.3
    u2 = ((xi[0] * eta + yi[0] * ksi) / (x - y)) ** 2
    w2 = u2 - c112 * b[3] / 2.0 / d14
    # values of coefficients Q2 and P2 from equation 3.4
    q2 = w2 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    p2 = q2 + c442 * b[3] / 2.0 / d14
    # values of R-functions to check
    rc_function_value_uw = integrals.rc(u2, w2)
    rc_function_value_qp = integrals.rc(p2, q2)
    rf_function_value = integrals.rf(m2, lm2, lp2)
    rd_function_value = integrals.rd(m2, lm2, lp2)
    rj_function_value = integrals.rj(m2, lm2, lp2, wp2)
    return [rc_function_value_uw, rc_function_value_qp, rf_function_value, rd_function_value, rj_function_value]


def test_integrals_cubic_case_functions_ijnkc(p: list, a: list, b: list, fgh: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_functions_rcrfrdrj. '
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
    wp2 = m2 - b[0] * (c142 + math.sqrt(c112 * c442)) / d14
    # values of coefficients U2 and W2 from equation 3.3
    u2 = ((xi[0] * eta + yi[0] * ksi) / (x - y)) ** 2
    w2 = u2 - c112 * b[3] / 2.0 / d14
    # values of coefficients Q2 and P2 from equation 3.4
    q2 = w2 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    p2 = q2 + c442 * b[3] / 2.0 / d14
    # values of R-functions to check
    rc_function_value_uw = integrals.rc(u2, w2)
    rc_function_value_qp = integrals.rc(p2, q2)
    rf_function_value = integrals.rf(m2, lm2, lp2)
    rd_function_value = integrals.rd(m2, lm2, lp2)
    rj_function_value = integrals.rj(m2, lm2, lp2, wp2)
    # additional coefficient from equation 3.5
    rho = math.sqrt(2 * fgh[2] * c112) - beta1
    i1c = 4 * rf_function_value
    i2c = math.sqrt(2 * c112 / 9.0 / fgh[2]) * (4 * rho * rd_function_value - 6 * rf_function_value
                                                + 3.0 / math.sqrt(u2)) + 2 * xi[0] * yi[0] / math.sqrt(u2)
    i3c = (2 * math.sqrt(c112) / 3.0 / math.sqrt(c442)) * \
          ((-4 * b[0] / d14) * (c142 + math.sqrt(c112 * c442)) * rj_function_value
           - 6 * rf_function_value + 3 * rc_function_value_uw) + 2 * rc_function_value_qp
    a111 = test_integrals_cubic_case_a_p1_pn(p[0], a, b, fgh, y, x)
    am111m2 = test_integrals_cubic_case_a_p1_pn(p[1], a, b, fgh, y, x)
    j1c = c112 * i1c / 2.0 - 2 * b[0] * a111
    n2c = math.sqrt(8 * fgh[2] / 9.0 / c112) * (4 * rho * rd_function_value - 6 * rf_function_value
                                                + 3.0 / math.sqrt(u2)) + 2.0 / (xi[0] * yi[0] * math.sqrt(u2))
    k2c = c112 * n2c / 2.0 - 2 * d14 * am111m2
    return [i1c, i2c, i3c, j1c, n2c, k2c]


def test_integrals_cubic_case_lmpwp(a: list, b: list, fgh: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_lmpwp. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficient d14 from equation 2.1
    d14 = a[0] * b[3] - a[3] * b[0]
    if d14 == 0.0:
        raise Exception('Error occurred in function test_integrals_cubic_case_lmpwp. '
                        'We have situation where d14 == 0.')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.1
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
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
    wp2 = m2 - b[0] * (c142 + math.sqrt(c112 * c442)) / d14
    return [lm2, lp2, wp2]


def test_integrals_cubic_case_m2(a: list, b: list, fgh: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_m2. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    xi = []
    yi = []
    alpha = []
    beta = []
    # list of coefficients xi and yi from equation 2.1 and coefficients alfa[i] and beta[i] from equation 2.2
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
        alpha.append(2 * fgh[0] * b[i] - fgh[1] * a[i])
        beta.append(fgh[1] * b[i] - 2 * fgh[2] * a[i])
    d2 = 4 * fgh[0] * fgh[2] - fgh[1] * fgh[1]
    if d2 <= 0.0:
        raise Exception('Error occurred in function test_integrals_cubic_case_m2. '
                        'We have situation where d2 is lower than 0, but should be greater than 0')
    cij = []
    for i in range(len(a)):
        for j in range(len(a)):
            cij.append(2 * fgh[0] * b[i] * b[j] - fgh[1] * (a[i] * b[j] + a[j] * b[i]) + 2 * fgh[2] * a[i] * a[j])
    # print(len(cij), cij)
    # coefficients ksi and eta given from equation 2.4
    ksi = math.sqrt(fgh[0] + fgh[1] * x + fgh[2] * x * x)
    eta = math.sqrt(fgh[0] + fgh[1] * y + fgh[2] * y * y)
    # return the value of M2 from equation 3.1
    return ((xi[0] + yi[0]) * math.sqrt((ksi + eta) ** 2 - fgh[2] * (x - y) ** 2) / (x - y)) ** 2


def test_integrals_cubic_case_uw(a: list, b: list, fgh: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_uw. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficient d14 from equation 2.1
    d14 = a[0] * b[3] - a[3] * b[0]
    if d14 == 0.0:
        raise Exception('Error occurred in function test_integrals_cubic_case_uw. We have situation where d14 <= 0.')
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.1
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    # coefficients c11, c14, c44 from equation 2.3
    c112 = compute_cij_value(a, b, fgh, 0, 0)
    # coefficients ksi and eta given from equation 2.4
    ksi = math.sqrt(fgh[0] + fgh[1] * x + fgh[2] * x * x)
    eta = math.sqrt(fgh[0] + fgh[1] * y + fgh[2] * y * y)
    # values of coefficients U2 and W2 from equation 3.3
    u2 = ((xi[0] * eta + yi[0] * ksi) / (x - y)) ** 2
    w2 = u2 - c112 * b[3] / 2.0 / d14
    return [u2, w2]


if __name__ == "__main__":
    start = time.time()
    p_test, a_test, b_test, fgh_test, y_test, x_test = cubic_cases_test_data()
    data_to_compare_with = carlson_intermediate_values_cubic_case()
    result_m2 = test_integrals_cubic_case_m2(a_test, b_test, fgh_test, y_test, x_test)
    if comparison_result_floats(result_m2, data_to_compare_with[0], eps=1e-6):
        print(f'Correct value of: M2 with eps = {1e-6}')
    result_lm2lp2wp2 = test_integrals_cubic_case_lmpwp(a_test, b_test, fgh_test, y_test, x_test)
    if comparison_result_lists(result_lm2lp2wp2, [data_to_compare_with[1], data_to_compare_with[2],
                                                  data_to_compare_with[3]], eps=1e-6):
        print(f'Correct values of: Lm2, Lp2, Wp2 with eps = {1e-6}')
    result_uw = test_integrals_cubic_case_uw(a_test, b_test, fgh_test, y_test, x_test)
    if comparison_result_lists(result_uw, [data_to_compare_with[4], data_to_compare_with[5]], eps=1e-6):
        print(f'Correct values of: U2, W2 with eps = {1e-6}')
    result_functions_rcrfrdrj = test_integrals_cubic_case_functions_rcrfrdrj(a_test, b_test, fgh_test, y_test, x_test)
    if comparison_result_lists(result_functions_rcrfrdrj, [data_to_compare_with[6], data_to_compare_with[7],
                                                           data_to_compare_with[8], data_to_compare_with[9],
                                                           data_to_compare_with[10]], eps=1e-5):
        print(f'Correct values of functions: RC, RF, RD, RJ with eps = {1e-5}')
    result_a_p1_pn = []
    list_of_data_to_compare_with = []
    for i in range(5):
        result_a_p1_pn.append(test_integrals_cubic_case_a_p1_pn(p_test[i + 1], a_test, b_test, fgh_test,
                                                                y_test, x_test))
        list_of_data_to_compare_with.append(data_to_compare_with[i + 18])
    if comparison_result_lists(result_a_p1_pn, list_of_data_to_compare_with, 1e-6):
        print(f'Correct values of A for different p[i] with eps = {1e-6}')
    result_ijnkc = test_integrals_cubic_case_functions_ijnkc([p_test[3], p_test[2]], a_test, b_test,
                                                             fgh_test, y_test, x_test)
    if comparison_result_lists(result_ijnkc, [data_to_compare_with[11], data_to_compare_with[12],
                                              data_to_compare_with[13], data_to_compare_with[14],
                                              data_to_compare_with[15], data_to_compare_with[16]], 1e-5):
        print(f'Correct values of I1c, I2c, I3c, J1c, N2c and K2c with eps = {1e-5}')
    koniec = time.time()
    time = (koniec - start) * 1000
    print(f'Tests took: {time} ms')
