from integrals import integrals
import math
import time


def carlson_intermediate_values_quartic_case() -> list:
    # Function which return a list of some intermediate values for given input
    # ai = 0.1 + 0.2i, bi = 0.5 - 0.2i (1<=i<=4), x = 2.0, y = 0.5
    u12 = 0.24791575    # index 0
    u13 = 0.20471575    # index 1
    u14 = 0.19031575    # index 2
    w22 = -0.35688425   # index 3
    w122 = 0.21911575   # index 4
    q22 = -3.20755523   # index 5
    q122 = 0.54102655   # index 6
    p22 = 3.0168477     # index 7
    p122 = 0.55102655   # index 8
    rcp22q22 = 0.34465425   # index 9
    rcp122q122 = 1.3553823  # index 10
    rfu12u13u14 = 2.1642326     # index 11
    rdu12u13u14 = 10.860876     # index 12
    rju12u13u14w22 = -6.4342997     # index 13
    rju12u13u14w122 = 9.9863171     # index 14
    i1 = 4.3284652      # index 15
    i2 = 6.3592902      # index 16
    i3 = 1.4305398      # index 17
    i3p = 2.9408494     # index 18
    a111m1 = 0.56155363     # index 19
    a1111 = -0.039947562    # index 20
    a111m3 = 2.7981283      # index 21
    a11m1m1 = 1.3368648     # index 22
    a111m1m2 = 0.0096998758     # index 23
    a1111m2 = -0.15740823   # index 24
    a3111 = 0.12035743      # index 25
    result_list = [u12, u13, u14, w22, w122, q22, q122, p22, p122, rcp22q22, rcp122q122, rfu12u13u14, rdu12u13u14,
                   rju12u13u14w22, rju12u13u14w122, i1, i2, i3, i3p, a111m1, a1111, a111m3, a11m1m1, a111m1m2,
                   a1111m2, a3111]
    return result_list


def comparison_result(list1: list, list2: list, eps: float) -> bool:
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


def quartic_all_real_cases_test_data() -> tuple:
    a_list = [0.3, 0.5, 0.7, 0.9, 0.8]
    b_list = [0.3, 0.1, -0.1, -0.3, 1.0]
    y_val = 0.5
    x_val = 2.0
    p_list = [[-1, -1, -1, -1, 0], [1, 1, 1, -1, 0], [1, 1, 1, 1, 0], [1, 1, 1, -3, 0], [1, 1, -1, -1, 0],
              [1, 1, 1, -1, -2], [1, 1, 1, 1, -2], [3, 1, 1, 1, 0]]
    # p_list is a list of lists
    # zero could be skipped but I will leave it to make sure every list has the same length
    # by default we will use p_list[0] = [ -1, -1, -1, -1, 0]
    # [1, 1, 1, -1, 0]      index 1
    # [1, 1, 1, 1, 0]       index 2
    # [1, 1, 1, -3, 0]      index 3
    # [1, 1, -1, -1, 0]     index 4
    # [1, 1, 1, -1, -2]     index 5
    # [1, 1, 1, 1, -2]      index 6
    # [3, 1, 1, 1, 0]       index 7
    return p_list, a_list, b_list, y_val, x_val


def test_integrals_quartic_all_real_case_a_p1_pn(p: list, a: list, b: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_all_real_case_a_p1_pn. '
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


def test_integrals_quartic_all_real_case_i1_i3(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_all_real_case_i1_i3. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficients d_{i,j} from equation 2.1
    d12 = a[0] * b[1] - a[1] * b[0]
    d13 = a[0] * b[2] - a[2] * b[0]
    d14 = a[0] * b[3] - a[3] * b[0]
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
    if a[4] == 1.0 and b[4] == 0.0:
        w122 = u12 * u12 - b[1] * d13 * d14 / b[0]
        q122 = w122 / (xi[0] * yi[0]) ** 2
        p122 = q122 + b[1] * b[2] * b[3] / b[0]
        rc_function_value = integrals.rc(p122, q122)
        rf_function_value = integrals.rf(u12 * u12, u13 * u13, u14 * u14)
        rd_function_value = integrals.rd(u12 * u12, u13 * u13, u14 * u14)
        rj_function_value = integrals.rj(u12 * u12, u13 * u13, u14 * u14, w122)
        i1 = 2.0 * rf_function_value
        i2 = 2.0 * d12 * d13 * rd_function_value / 3.0 + 2.0 * xi[0] * yi[0] / xi[3] / yi[3] / u14
        i3p = -2.0 * d12 * d13 * d14 * rj_function_value / 3.0 / b[0] + 2.0 * rc_function_value
        return [i1, i2, i3p]
    else:
        d15 = a[0] * b[4] - a[4] * b[0]
        d25 = a[1] * b[4] - a[4] * b[1]
        d35 = a[2] * b[4] - a[4] * b[2]
        d45 = a[3] * b[4] - a[4] * b[3]
        w22 = u12 * u12 - d13 * d14 * d25 / d15
        q22 = w22 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p22 = q22 + d25 * d35 * d45 / d15
        rc_function_value = integrals.rc(p22, q22)
        rf_function_value = integrals.rf(u12 * u12, u13 * u13, u14 * u14)
        rd_function_value = integrals.rd(u12 * u12, u13 * u13, u14 * u14)
        rj_function_value = integrals.rj(u12 * u12, u13 * u13, u14 * u14, w22)
        i1 = 2.0 * rf_function_value
        i2 = 2.0 * d12 * d13 * rd_function_value / 3.0 + 2.0 * xi[0] * yi[0] / xi[3] / yi[3] / u14
        i3 = 2.0 * d12 * d13 * d14 * rj_function_value / 3.0 / d15 + 2.0 * rc_function_value
        return [i1, i2, i3]


def test_integrals_quartic_all_real_case_rcrfrdrj(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_all_real_case_rcrfrdrj. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficients d_{i,j} from equation 2.1
    d13 = a[0] * b[2] - a[2] * b[0]
    d14 = a[0] * b[3] - a[3] * b[0]
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
    if a[4] == 1.0 and b[4] == 0.0:
        w122 = u12 * u12 - b[1] * d13 * d14 / b[0]
        q122 = w122 / (xi[0] * yi[0]) ** 2
        p122 = q122 + b[1] * b[2] * b[3] / b[0]
        rc_function_value = integrals.rc(p122, q122)
        rf_function_value = integrals.rf(u12 * u12, u13 * u13, u14 * u14)
        rd_function_value = integrals.rd(u12 * u12, u13 * u13, u14 * u14)
        rj_function_value = integrals.rj(u12 * u12, u13 * u13, u14 * u14, w122)
        return [rc_function_value, rf_function_value, rd_function_value, rj_function_value]
    else:
        d15 = a[0] * b[4] - a[4] * b[0]
        d25 = a[1] * b[4] - a[4] * b[1]
        d35 = a[2] * b[4] - a[4] * b[2]
        d45 = a[3] * b[4] - a[4] * b[3]
        w22 = u12 * u12 - d13 * d14 * d25 / d15
        q22 = w22 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p22 = q22 + d25 * d35 * d45 / d15
        rc_function_value = integrals.rc(p22, q22)
        rf_function_value = integrals.rf(u12 * u12, u13 * u13, u14 * u14)
        rd_function_value = integrals.rd(u12 * u12, u13 * u13, u14 * u14)
        rj_function_value = integrals.rj(u12 * u12, u13 * u13, u14 * u14, w22)
        return [rc_function_value, rf_function_value, rd_function_value, rj_function_value]


def test_integrals_quartic_all_real_case_ui(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_all_real_case_ui. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
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
    return [u12 * u12, u13 * u13, u14 * u14]


def test_integrals_quartic_all_real_case_wqp(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_quartic_all_real_case_wqp. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
    # coefficients d_{i,j} from equation 2.1
    d13 = a[0] * b[2] - a[2] * b[0]
    d14 = a[0] * b[3] - a[3] * b[0]
    xi = []
    yi = []
    # list of coefficients xi and yi from equation 2.2
    for i in range(len(a)):
        xi.append(math.sqrt(a[i] + b[i] * x))
        yi.append(math.sqrt(a[i] + b[i] * y))
    # coefficients ui from equation 2.3
    u12 = (xi[0] * xi[1] * yi[2] * yi[3] + yi[0] * yi[1] * xi[2] * xi[3]) / (x - y)
    if a[4] == 1.0 and b[4] == 0.0:
        w122 = u12 * u12 - b[1] * d13 * d14 / b[0]
        q122 = w122 / (xi[0] * yi[0]) ** 2
        p122 = q122 + b[1] * b[2] * b[3] / b[0]
        return [w122, q122, p122]
    else:
        d15 = a[0] * b[4] - a[4] * b[0]
        d25 = a[1] * b[4] - a[4] * b[1]
        d35 = a[2] * b[4] - a[4] * b[2]
        d45 = a[3] * b[4] - a[4] * b[3]
        w22 = u12 * u12 - d13 * d14 * d25 / d15
        q22 = w22 * (xi[4] * yi[4] / xi[0] / yi[0]) ** 2
        p22 = q22 + d25 * d35 * d45 / d15
        return [w22, q22, p22]


if __name__ == "__main__":
    start = time.time()
    p_test, a_test, b_test, y_test, x_test = quartic_all_real_cases_test_data()
    result_ui = test_integrals_quartic_all_real_case_ui(a_test, b_test, y_test, x_test)
    data_to_compare_with = carlson_intermediate_values_quartic_case()
    if comparison_result(result_ui, [data_to_compare_with[0], data_to_compare_with[1], data_to_compare_with[2]], 1e-6):
        print("Correct values of: U12, U22, U32 with eps = 1e-6")
    result_wqp = test_integrals_quartic_all_real_case_wqp(a_test, b_test, y_test, x_test)
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result(result_wqp, [data_to_compare_with[4], data_to_compare_with[6], data_to_compare_with[8]],
                             1e-5):
            print("Correct values of: W122, Q122, P122 with eps = 1e-5")
    else:
        if comparison_result(result_wqp, [data_to_compare_with[3], data_to_compare_with[5], data_to_compare_with[7]],
                             1e-5):
            print("Correct values of: W22, Q22, P22 with eps = 1e-5")
    result_rcrfrdrj = test_integrals_quartic_all_real_case_rcrfrdrj(a_test, b_test, y_test, x_test)
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result(result_rcrfrdrj, [data_to_compare_with[10], data_to_compare_with[11],
                                               data_to_compare_with[12], data_to_compare_with[14]], 1e-4):
            print("Correct values of: RC, RF, RD, RJ functions with eps = 1e-6")
    else:
        if comparison_result(result_rcrfrdrj, [data_to_compare_with[9], data_to_compare_with[11],
                                               data_to_compare_with[12], data_to_compare_with[13]], 1e-4):
            print("Correct values of: RC, RF, RD, RJ functions with eps = 1e-6")
    result_i1_i3 = test_integrals_quartic_all_real_case_i1_i3(a_test, b_test, y_test, x_test)
    if a_test[4] == 1.0 and b_test[4] == 0.0:
        if comparison_result(result_i1_i3, [data_to_compare_with[15], data_to_compare_with[16],
                                               data_to_compare_with[18]], 1e-4):
            print("Correct values of: I1, I2, I3 prime with eps = 1e-6")
    else:
        if comparison_result(result_i1_i3, [data_to_compare_with[15], data_to_compare_with[16],
                                               data_to_compare_with[17]], 1e-4):
            print("Correct values of: I1, I2, I3 with eps = 1e-6")
    result_a_p1_pn = []
    list_of_data_to_compare_with = []
    for i in range(6):
        result_a_p1_pn.append(test_integrals_quartic_all_real_case_a_p1_pn(p_test[i + 1], a_test,
                                                                           b_test, y_test, x_test))
        list_of_data_to_compare_with.append(data_to_compare_with[i + 19])
    if comparison_result(result_a_p1_pn, list_of_data_to_compare_with, 1e-6):
        print("Correct values of A for different p[i] with eps = 1e-6")
    koniec = time.time()
    time = (koniec - start) * 1000
    print(f'Tests took: {time} ms')
