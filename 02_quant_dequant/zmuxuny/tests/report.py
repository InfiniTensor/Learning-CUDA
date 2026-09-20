"""Rebuild final performance tables from committed experiment records."""

from report_final import ROOT, summary, finalize_platform_report
from report_4090 import main as nvidia_report

if __name__ == "__main__":
    summary(ROOT)
    nvidia_report()
    for platform in ["c500", "iluvatar", "musa", "ascend"]:
        finalize_platform_report(ROOT, platform)
