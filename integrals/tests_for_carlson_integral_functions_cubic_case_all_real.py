from integrals import integrals
import math
import time


def carlson_intermediate_values_cubic_case() -> list:
    # Function which return a list of some intermediate values for given input
    # ai = 0.1 + 0.2i, bi = 0.5 - 0.2i (1<=i<=4), x = 2.0, y = 0.5
    u12 = 0.41309998    # index 0
    u22 = 0.40109998    # index 1
    u32 = 0.43709998    # index 2
    w22 = 0.38909998    # index 3
    q22 = 0.21616665    # index 4
    p22 = 0.24016665    # index 5
    rcp22q22 = 2.1128946    # index 6
    rfu32u22u12 = 1.5486858     # index 7
    rdu32u22u12 = 3.7353179     # index 8
    rju32u22u12w22 = 3.8709720  # index 9
    i1c = 3.0973715     # index 10
    i2c = 2.0520132     # index 11
    i3c = 4.2877248     # index 12
    j1c = -0.00688951   # index 13
    j2c = -0.80566308   # index 14
    k2c = 0.78110328    # index 15
    a111 = 0.16015635   # index 16
    a11m1 = 0.50543220  # index 17
    a1m1m1 = 0.48163106     # index 18
    am11m1 = -0.12403646    # index 19
    a11m3 = 1.2956636   # index 20
    a311 = 0.32463223   # index 21
    a111m2 = 1.3360390  # index 22
    a11m1m2 = 2.9189040     # index 23
    result_list = [u12, u22, u32, w22, q22, p22, rcp22q22, rfu32u22u12, rdu32u22u12, rju32u22u12w22, i1c, i2c, i3c,
                   j1c, j2c, k2c, a111, a11m1, a1m1m1, am11m1, a11m3, a311, a111m2, a11m1m2]
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


def cubic_cases_test_data() -> tuple:
    a_list = [0.3, 0.5, 0.7, 0.9]
    b_list = [0.3, 0.1, -0.1, -0.3]
    y_val = 0.5
    x_val = 2.0
    p_list = [[-1, -1, -1, 0], [1, 1, 1, 0], [1, 1, -1, 0], [1, -1, -1, 0], [-1, 1, -1, 0], [1, 1, -3, 0], [3, 1, 1, 0],
              [1, 1, 1, -2], [1, 1, -1, -2]]
    # p_list is a list of lists
    # zero could be skipped but I will leave it to make sure every list has the same length
    # by default we will use p_list[0] = [ -1, -1, -1, 0]
    # [1, 1, 1, 0]      index 1
    # [1, 1, -1, 0]     index 2
    # [1, -1, -1, 0]    index 3
    # [-1, 1, -1, 0]    index 4
    # [1, 1, -3, 0]     index 5
    # [3, 1, 1, 0]      index 6
    # [1, 1, 1, -2]     index 7
    # [1, 1, -1, -2]    index 8
    return p_list, a_list, b_list, y_val, x_val


def test_integrals_cubic_case_a_p1_pn(p: list, a: list, b: list, y: float, x: float) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_a_p1_pn. '
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


def test_integrals_cubic_case_ui(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_ui. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
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
    return [u1 * u1, u2 * u2, u3 * u3]


def test_integrals_cubic_case_functions_rcrfrdrj(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_functions_rcrfrdrj. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
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
    # coefficients w22 from equation 2.4
    w22 = u1 * u1 - b[3] * d12 * d13 / d14
    # coefficients q22 and p22 from equation 2.5
    q22 = w22 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    p22 = q22 + b[3] * d24 * d34 / d14
    rc_function_value = integrals.rc(p22, q22)
    rf_function_value = integrals.rf(u3 * u3, u2 * u2, u1 * u1)
    rd_function_value = integrals.rd(u3 * u3, u2 * u2, u1 * u1)
    rj_function_value = integrals.rj(u3 * u3, u2 * u2, u1 * u1, w22)
    return [rc_function_value, rf_function_value, rd_function_value, rj_function_value]


def test_integrals_cubic_case_i123c(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_i123c. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
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
    # coefficients w22 from equation 2.4
    w22 = u1 * u1 - b[3] * d12 * d13 / d14
    # coefficients q22 and p22 from equation 2.5
    q22 = w22 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    p22 = q22 + b[3] * d24 * d34 / d14
    i1c = 2.0 * integrals.rf(u3 * u3, u2 * u2, u1 * u1)
    i2c = 2.0 * d12 * d13 * integrals.rd(u3 * u3, u2 * u2, u1 * u1) / 3.0 + 2.0 * xi[0] * yi[0] / u1
    i3c = -2.0 * b[0] * d12 * d13 * integrals.rj(u3 * u3, u2 * u2, u1 * u1, w22) / 3.0 / d14 + 2.0 * \
          integrals.rc(p22, q22)
    return [i1c, i2c, i3c]


def test_integrals_cubic_case_jk_values(p: list, a: list, b: list, y: float, x: float,
                                        i1: bool, i2: bool, j1: bool, j2: bool, k2: bool) -> float:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_jk_values. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
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
    # coefficients w22 from equation 2.4
    w22 = u1 * u1 - b[3] * d12 * d13 / d14
    # coefficients q22 and p22 from equation 2.5
    q22 = w22 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    ax = 1
    ay = 1
    # coefficients A from equation 2.6
    for j in range(len(a)):
        ax = ax * xi[j] ** p[j]
        ay = ay * yi[j] ** p[j]
    axy = ax - ay
    if i1:
        i1c = 2.0 * integrals.rf(u3 * u3, u2 * u2, u1 * u1)
    if i2:
        i2c = 2.0 * d12 * d13 * integrals.rd(u3 * u3, u2 * u2, u1 * u1) / 3.0 + 2.0 * xi[0] * yi[0] / u1
    if j1:
        j1c = d12 * d13 * i1c - 2.0 * b[0] * axy
        return j1c
    if j2:
        j2c = b[1] * i2c - 2.0 * axy
        return j2c
    if k2:
        k2c = b[1] * b[2] * i2c - 2.0 * b[3] * axy
        return k2c


def test_integrals_cubic_case_wpq_coefficients(a: list, b: list, y: float, x: float) -> list:
    if x < y:
        raise Exception('Error occurred in function test_integrals_cubic_case_wqp_coefficients. '
                        'We have situation where x is lower than y. The value of x must be greater than y')
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
    # coefficients w22 from equation 2.4
    w22 = u1 * u1 - b[3] * d12 * d13 / d14
    # coefficients q22 and p22 from equation 2.5
    q22 = w22 * (xi[3] * yi[3] / xi[0] / yi[0]) ** 2
    p22 = q22 + b[3] * d24 * d34 / d14
    return [w22, q22, p22]


if __name__ == "__main__":
    start = time.time()
    p_test, a_test, b_test, y_test, x_test = cubic_cases_test_data()
    result_ui = test_integrals_cubic_case_ui(a_test, b_test, y_test, x_test)
    data_to_compare_with = carlson_intermediate_values_cubic_case()
    if comparison_result(result_ui, [data_to_compare_with[0], data_to_compare_with[1], data_to_compare_with[2]], 1e-6):
        print("Correct values of: U12, U22, U32 with eps = 1e-6")
    result_wqp = test_integrals_cubic_case_wpq_coefficients(a_test, b_test, y_test, x_test)
    if comparison_result(result_wqp, [data_to_compare_with[3], data_to_compare_with[4], data_to_compare_with[5]], 1e-6):
        print("Correct values of: W22, Q22, P22 with eps = 1e-6")
    result_r_functions = test_integrals_cubic_case_functions_rcrfrdrj(a_test, b_test, y_test, x_test)
    if comparison_result(result_r_functions, [data_to_compare_with[6], data_to_compare_with[7],
                                              data_to_compare_with[8], data_to_compare_with[9]], 1e-6):
        print("Correct values of RC, RF, RD and RJ functions with eps = 1e-6")
    result_a_p1_pn = []
    list_of_data_to_compare_with = []
    for i in range(8):
        result_a_p1_pn.append(test_integrals_cubic_case_a_p1_pn(p_test[i + 1], a_test, b_test, y_test, x_test))
        list_of_data_to_compare_with.append(data_to_compare_with[i + 16])
    if comparison_result(result_a_p1_pn, list_of_data_to_compare_with, 1e-6):
        print("Correct values of A for different p[i] with eps = 1e-6")
    result_i123c = test_integrals_cubic_case_i123c(a_test, b_test, y_test, x_test)
    if comparison_result(result_i123c, [data_to_compare_with[10], data_to_compare_with[11],
                                        data_to_compare_with[12]], 1e-6):
        print("Correct values of: I1c, I2c, I3c with eps = 1e-6")
    result_j1c = test_integrals_cubic_case_jk_values(p_test[1], a_test, b_test, y_test, x_test, True,
                                                     False, True, False, False)
    result_j2c = test_integrals_cubic_case_jk_values(p_test[2], a_test, b_test, y_test, x_test, False,
                                                     True, False, True, False)
    result_k2c = test_integrals_cubic_case_jk_values(p_test[7], a_test, b_test, y_test, x_test, False,
                                                     True, False, False, True)
    if comparison_result([result_j1c, result_j2c, result_k2c],
                         [data_to_compare_with[13], data_to_compare_with[14], data_to_compare_with[15]], 1e-6):
        print("Correct values of: J1c, J2c, K2c with eps = 1e-6")
    koniec = time.time()
    time = (koniec - start) * 1000
    print(f'Tests took: {time} ms')

