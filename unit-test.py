import unittest
import logging
import coverage
import sys

cov = coverage.Coverage()
cov.start()

from lycoris.logging import logger

logger.setLevel(logging.ERROR)

def _import_test_class(module_name: str, class_name: str):
    """Import a test class lazily; returns None when optional deps are missing."""
    try:
        mod = __import__(module_name, fromlist=[class_name])
        return getattr(mod, class_name)
    except ImportError as e:
        print(f"SKIPPING {module_name}.{class_name}: import error - {e}")
        return None


_test_specs = [
    ("test.module", "LycorisModuleTests"),
    ("test.wrapper", "LycorisWrapperTests"),
    ("test.functional", "LycorisFunctionalTests"),
    ("test.kohya", "LycorisKohyaWrapperTests"),
    ("test.precision_merge_test", "MergePrecisionTests"),
    ("test.kernels.test_ops", "OpsVsFp64"),
    ("test.kernels.test_autograd", "AutogradParity"),
    ("test.kernels.test_autograd", "SafeFallback"),
]

# Test classes that imported successfully. The kernel suites skip
# themselves without CUDA.
TESTS = [tc for tc in (_import_test_class(m, c) for m, c in _test_specs) if tc is not None]

if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    all_suites = [test_loader.loadTestsFromTestCase(t) for t in TESTS]

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
