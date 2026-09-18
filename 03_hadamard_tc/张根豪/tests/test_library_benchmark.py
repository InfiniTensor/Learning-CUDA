import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summary_library',ROOT/'tensor_core/scripts/summarize_library.py')
SUMMARY=importlib.util.module_from_spec(spec); spec.loader.exec_module(SUMMARY)


class LibraryBenchmarkTest(unittest.TestCase):
    def test_guard_fallback_butterfly_layout(self):
        for dim in (64,128,256):
            elems=dim//32
            expected=[float((i*17)%31-15) for i in range(dim)]
            lanes=[expected[lane*elems:(lane+1)*elems] for lane in range(32)]
            stride=1
            while stride<dim:
                old=expected[:]
                expected=[old[i]+old[i^stride] if not i&stride else old[i^stride]-old[i] for i in range(dim)]
                stride*=2
            for values in lanes:
                stride=1
                while stride<elems:
                    for j in range(elems):
                        if not j&stride:
                            a,b=values[j],values[j|stride]
                            values[j],values[j|stride]=a+b,a-b
                    stride*=2
            offset=1
            while offset<32:
                old=[v[:] for v in lanes]
                for lane in range(32):
                    for j in range(elems):
                        lanes[lane][j]=old[lane^offset][j]-old[lane][j] if lane&offset else old[lane][j]+old[lane^offset][j]
                offset*=2
            self.assertEqual([v for lane in lanes for v in lane],expected)

    def test_ptx_c_to_b_shuffle_mapping(self):
        # Symbolic matrix entries validate every destination lane/fragment element.
        for lane in range(32):
            g,t=lane>>2,lane&3
            for half in (0,1):
                for reg in (0,1):
                    for j in (0,1):
                        source=4*(2*t+j)+g//2
                        ci=2*reg+(g&1)
                        actual=(source//4+(8 if ci>=2 else 0),2*(source%4)+(ci&1)+8*half)
                        self.assertEqual(actual,(2*t+j+8*reg,g+8*half))

    def test_median_and_failed_validation_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'timing.csv'
            row=dict(gpu='test',dtype='float16',tokens='1',head_dim='64',normalize='1',backend='warp',measurement='graph_amortized_device')
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row)+['round','us','pass']); w.writeheader()
                for n,t in enumerate((3,1,2)): w.writerow(dict(row,round=n,us=t,**{'pass':n!=0}))
            result=SUMMARY.aggregate(path)[0]
            self.assertEqual(result['median_us'],2)
            self.assertEqual(result['repeats'],3)
            self.assertFalse(result['all_pass'])
            with path.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row)+['round','us','pass'])
                w.writerow(dict(row,gpu='other',round=0,us=1,**{'pass':True}))
            with self.assertRaises(ValueError): SUMMARY.aggregate(path)


if __name__=='__main__': unittest.main()
