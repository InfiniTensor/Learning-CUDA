"""CPU-only regression tests for strict-audit tooling and units."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tensor_core/scripts' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


AUDIT = module('summarize_audit')
VALIDATE = module('validate_tc')


class AuditToolsTest(unittest.TestCase):
    def test_full_matrix_covers_both_dtypes_normalizations_and_tails(self):
        cases = list(VALIDATE.cases())
        self.assertEqual(len(cases), 132)
        self.assertEqual(len(set(cases)), 132)
        for dtype in ('fp16', 'bf16'):
            for dim in (64, 128, 256, 512, 1024):
                for norm in (True, False):
                    for tokens in (1, 5, 2048):
                        self.assertIn((dtype, dim, norm, tokens, 'normal'), cases)

    def test_wide_csv_uses_units_and_rejects_multiple_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ncu.csv'
            with path.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([
                    ['gpu__time_duration.avg', 'dram__bytes.sum.per_second'],
                    ['us', 'Gbyte/s'], ['5', '800']])
            metrics = AUDIT.ncu_metrics(path)
            duration = AUDIT.metric(metrics, 'gpu__time_duration.avg', 'seconds')
            rate = AUDIT.metric(metrics, 'dram__bytes.sum.per_second', 'bytes/s')
            self.assertAlmostEqual(duration, 5e-6)
            self.assertAlmostEqual(rate * duration, 4e6)
            with path.open('a') as f: f.write('6,900\n')
            with self.assertRaises(ValueError): AUDIT.ncu_metrics(path)

    def test_missing_and_invalid_metrics_are_not_fabricated(self):
        self.assertIsNone(AUDIT.metric({}, 'missing'))
        self.assertIsNone(AUDIT.metric({'a': ('nan', 'us')}, 'a', 'seconds'))
        with self.assertRaises(ValueError):
            AUDIT.metric({'a': ('1', 'unknown')}, 'a', 'seconds')

    def test_profile_dimension_is_not_hardcoded(self):
        self.assertEqual(AUDIT.profile_case('warp_d64_all_raw'), ('warp', 64, 'all'))
        self.assertEqual(AUDIT.profile_case('tc_split_none_raw'), ('tc_split', 128, 'none'))
        with self.assertRaises(ValueError):
            AUDIT.profile_case('warp_d123_all_raw')

    def test_native_csv_has_no_relative_error_acceptance(self):
        # Guard the specific historical regression: PASS(rel) bypassing PDF.
        code = (ROOT / 'tensor_core/src/tc_bench.cu').read_text()
        self.assertNotIn('return acc.max_abs_error < threshold ||', code)
        self.assertIn('return acc.reference_max_abs_error < threshold;', code)
        self.assertNotIn('PASS(rel)', code)


if __name__ == '__main__':
    unittest.main()
