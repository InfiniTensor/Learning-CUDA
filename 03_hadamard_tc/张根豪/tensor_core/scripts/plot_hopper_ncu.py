#!/usr/bin/env python3
"""Render NCU evidence independently of later benchmark jobs."""
import argparse
from pathlib import Path
from summarize_hopper import counters, save_csv
from roofline_plot import plot_roofline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--dram-peak-gbs',type=float,required=True)
    p.add_argument('--fp32-peak-tflops',type=float,required=True)
    args=p.parse_args();data=counters(args.directory)
    if len(data)!=30: raise ValueError('expected all 30 NCU captures')
    save_csv(args.directory/'ncu_summary.csv',data)
    plot_roofline(data, args.directory/'roofline.png', args.dram_peak_gbs, args.fp32_peak_tflops)


if __name__=='__main__': main()
