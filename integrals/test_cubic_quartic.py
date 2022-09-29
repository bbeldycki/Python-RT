from integrals import integrals
import time


def comparison_result_floats(value1: float, value2: float, eps: float) -> bool:
    # Function to compare my computed values with values from Carlson computations
    if abs(value1 - value2) > eps:
        raise Exception('Error occurred during comparison of ours computed results with values from carlson '
                        f'article. Reason: Absolute value of difference is larger than {eps}.')
    return True


def expected_values() -> list:
    int_cubic_all_real_roots = 1.5486858
    int_cubic_one_real_two_complex_roots = 0.89978529
    int_quartic_all_real_roots = 2.1642326
    int_quartic_two_real_two_complex_roots = 1.2543726
    int_quartic_all_complex_roots = 0.54784092
    return [int_cubic_all_real_roots, int_cubic_one_real_two_complex_roots, int_quartic_all_real_roots,
            int_quartic_two_real_two_complex_roots, int_quartic_all_complex_roots]


def test_input_data_cubic_case_all_real() -> tuple:
    a_list = [0.3, 0.5, 0.7, 0.9]
    b_list = [0.3, 0.1, -0.1, -0.3]
    y_val = 0.5
    x_val = 2.0
    p_list = [-1, -1, -1, 0]
    return p_list, a_list, b_list, y_val, x_val


def test_input_data_cubic_case_one_real_two_complex() -> tuple:
    a_list = [0.3, 0.0, 0.0, 0.9, 0.4]
    b_list = [0.2, 0.0, 0.0, -0.3, 0.5]
    kwadrat_list = [0.4, -0.2, 0.1]
    y_val = 0.5
    x_val = 2.0
    p_list = [-1, -1, -1, 0, 0]
    return p_list, a_list, b_list, kwadrat_list, y_val, x_val


def test_input_data_quartic_case_all_complex() -> tuple:
    a_list = [0.0, 0.0, 0.0, 0.0, 1.1]
    b_list = [0.0, 0.0, 0.0, 0.0, -0.4]
    fgh_1_list = [2.7, -1.8, 0.9]
    fgh_2_list = [2.0, 2.4, 0.8]
    y_val = -3.0
    x_val = 2.0
    p_list = [-1, -1, -1, -1, 0]
    return p_list, a_list, b_list, fgh_1_list, fgh_2_list, y_val, x_val


def test_input_data_quartic_case_all_real() -> tuple:
    a_list = [0.3, 0.5, 0.7, 0.9, 0.8]
    b_list = [0.3, 0.1, -0.1, -0.3, 1.0]
    y_val = 0.5
    x_val = 2.0
    p_list = [-1, -1, -1, -1, 0]
    return p_list, a_list, b_list, y_val, x_val


def test_input_data_quartic_case_two_real_and_complex() -> tuple:
    a_list = [0.3, 0.0, 0.0, 0.9, 0.4]
    b_list = [0.2, 0.0, 0.0, -0.3, 0.5]
    kwadrat_list = [0.4, -0.2, 0.1]
    y_val = 0.5
    x_val = 2.0
    p_list = [-1, -1, -1, 0, 0]
    return p_list, a_list, b_list, kwadrat_list, y_val, x_val


if __name__ == "__main__":
    start = time.time()
    ffr = 1.0
    p_test, a_test, b_test, y_test, x_test = test_input_data_cubic_case_all_real()
    eps = 1e-6
    print(expected_values()[0])
    if comparison_result_floats(integrals.elliptical_integral_cubic_all_roots_real
                                    (p_test, a_test, b_test, ffr, y_test, x_test) / 2.0,
                                expected_values()[0], eps=eps):
        print(f'Cubic integral with all real roots computed correctly with accuracy = {eps}')
    p_test, a_test, b_test, fgh_test, y_test, x_test = test_input_data_cubic_case_one_real_two_complex()
    eps = 1e-6
    if comparison_result_floats(integrals.elliptical_integral_cubic_one_real_and_two_complex_roots
                                    (p_test, a_test, b_test, fgh_test, ffr, y_test, x_test) / 4.0,
                                expected_values()[1], eps=eps):
        print(f'Cubic integral with one real root and two complex computed correctly with accuracy = {eps}')
    p_test, a_test, b_test, y_test, x_test = test_input_data_quartic_case_all_real()
    eps = 1e-6
    if comparison_result_floats(integrals.elliptical_integral_quartic_all_real_roots
                                    (p_test, a_test, b_test, ffr, y_test, x_test) / 2.0,
                                expected_values()[2], eps=eps):
        print(f'Quartic integral with all real roots computed correctly with accuracy = {eps}')
    p_test, a_test, b_test, fgh_test, y_test, x_test = test_input_data_quartic_case_two_real_and_complex()
    eps = 1e-6
    if comparison_result_floats(integrals.elliptical_integral_quartic_two_real_and_two_complex_roots
                                    (p_test, a_test, b_test, fgh_test, ffr, y_test, x_test) / 4.0,
                                expected_values()[3], eps=eps):
        print(f'Quartic integral with two real and two complex roots computed correctly with accuracy = {eps}')
    p_test, a_test, b_test, fgh1_test, fgh2_test, y_test, x_test = test_input_data_quartic_case_all_complex()
    eps = 1e-6
    if comparison_result_floats(integrals.elliptical_integral_quartic_all_complex_roots
                                    (p_test, a_test, b_test, fgh1_test, fgh2_test, ffr, y_test, x_test) / 4.0,
                                expected_values()[4], eps=eps):
        print(f'Quartic integral with all complex roots computed correctly with accuracy = {eps}')
    koniec = time.time()
    time = (koniec - start) * 1000
    print(f'Tests took: {time} ms')
