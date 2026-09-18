import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tensor_core/scripts'))
from summarize_hopper import counters
from summarize_audit import metric
from roofline_plot import plot_roofline


class HopperReportTest(unittest.TestCase):
    def test_roofline_input_guards_without_plotting_dependencies(self):
        with self.assertRaises(ValueError): plot_roofline([],None,4800,67)
        with self.assertRaises(ValueError): plot_roofline([{'gpu':'a'},{'gpu':'b'}],None,4800,67)
        with self.assertRaises(ValueError): plot_roofline([{'gpu':'a'}],None,0,67)

    def test_shared_per_block_units(self):
        self.assertEqual(metric({'shared':('12.5','Kbyte/block')},'shared','bytes'),12500)

    def write_capture(self, directory, name, gpu='test'):
        rows=[{'device__attribute_display_name':'', 'gpu__time_duration.avg':'usecond',
               'dram__bytes.sum.per_second':'Gbyte/s'},
              {'device__attribute_display_name':gpu, 'gpu__time_duration.avg':'10',
               'dram__bytes.sum.per_second':'100'}]
        with (directory/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

    def test_rate_duration_and_useful_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'ncu').mkdir()
            self.write_capture(directory/'ncu','ptx_fast_d64_all_raw.csv')
            row=counters(directory)[0]
            self.assertAlmostEqual(row['dram_bytes'],1e6)
            self.assertAlmostEqual(row['useful_gflops'],16384*64*6/1e4)
            self.assertIsNone(row['dynamic_shared'])  # Missing never becomes fake zero.
            self.write_capture(directory/'ncu','warp_d64_all_raw.csv','other GPU')
            with self.assertRaises(ValueError): counters(directory)

    def test_unknown_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'ncu').mkdir()
            self.write_capture(directory/'ncu','ptx_fast_d1024_all_raw.csv')
            with self.assertRaises(ValueError): counters(directory)


if __name__=='__main__': unittest.main()
