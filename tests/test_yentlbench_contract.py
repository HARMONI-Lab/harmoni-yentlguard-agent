import unittest

class TestYentlbenchContract(unittest.TestCase):
    def test_no_load_vignettes_in_yentlbench_data(self):
        """
        CRITICAL: yentlbench.data does not exist.
        YentlGuard's CLI currently calls:
            from yentlbench.data import load_vignettes
        This will raise ModuleNotFoundError at runtime.
        This test documents the mismatch so it can be fixed.
        """
        with self.assertRaises((ModuleNotFoundError, ImportError)):
            from yentlbench.data import load_vignettes