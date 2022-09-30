from unittest import TestLoader, TextTestRunner, TestSuite
from unit_tests.functions_tests import TestFunctionMuRoots, TestFunctionPolynomialRoots
from unit_tests.functions_tests import TestFunctionNodesAndWeightsGaussLegendre
from unit_tests.integrals_tests import TestEllipticalIntegralsCubic, TestEllipticalIntegralsQuartic


if __name__ == '__main__':
    loader = TestLoader()

    functions_tests_list = [TestFunctionMuRoots, TestFunctionPolynomialRoots, TestFunctionNodesAndWeightsGaussLegendre]
    tests = [loader.loadTestsFromTestCase(test) for test in functions_tests_list]

    integrals_tests_list = [TestEllipticalIntegralsCubic, TestEllipticalIntegralsQuartic]
    for test in integrals_tests_list:
        tests.append(loader.loadTestsFromTestCase(test))

    suite = TestSuite(tests)
    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
