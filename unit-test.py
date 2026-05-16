import unittest
import logging
import coverage
import sys

cov = coverage.Coverage()
cov.start()

from lycoris.logging import logger

logger.setLevel(logging.ERROR)

TESTS = [
    ("test.module", "LycorisModuleTests"),
    ("test.functional", "LycorisFunctionalTests"),
    ("test.wrapper", "LycorisWrapperTests"),
    ("test.kohya", "LycorisKohyaWrapperTests"),
]

if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    all_suites = []
    for module_name, test_name in TESTS:
        try:
            mod = __import__(module_name, fromlist=[test_name])
            test_case = getattr(mod, test_name)
            suite = test_loader.loadTestsFromTestCase(test_case)
            all_suites.append(suite)
        except ImportError as e:
            print(f"SKIPPING {module_name}.{test_name}: import error - {e}")

    if not all_suites:
        print("No test suites could be loaded!")
        sys.exit(1)

    combined = unittest.TestSuite(all_suites)
    result = runner.run(combined)

    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory="coverage_report")
    if not result.wasSuccessful():
        sys.exit(1)
