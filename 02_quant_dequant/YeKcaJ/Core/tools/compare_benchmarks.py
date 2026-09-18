"""仅比较固定协议、相同输入哈希和相同统计口径的整组结果。"""
import argparse
import json
import os
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
INPUT_HASHES = json.loads((CORE / "configs/benchmark-inputs-v1.json").read_text())


def validate_pair(a, b):
    for env in (a, b):
        if env.get("protocol") != "fixed-fp32-v1" or env.get("formal") is not True:
            raise ValueError("requires formal fixed-fp32-v1 benchmark; historical/exploratory data excluded")
        expected_input = dict(dtype="fp32", shape="1 x elements", elements=[1 << 20, 4 << 20, 16 << 20],
                              generator_seed=20260909, sha256=INPUT_HASHES)
        if any(env.get("input", {}).get(k) != v for k, v in expected_input.items()):
            raise ValueError("input does not match frozen fixed-fp32-v1 corpus")
        if env.get("quantization") != dict(scale_mode="block", rounding="nearest", seed=1234,
                                            block_size=dict(mxfp8=32, nvfp4=16)):
            raise ValueError("quantization settings do not match protocol")
        if env.get("metric") != dict(op="quant_optimized", scope="resident_gpu", statistic="median_ms",
                                     p95="nearest-rank", output_types=["fp32", "fp16", "bf16"]):
            raise ValueError("timing metric does not match protocol")
    for key in ("protocol", "input", "quantization", "metric", "warmups", "repeats"):
        if a[key] != b[key]:
            raise ValueError(f"benchmark conditions differ: {key}")
    if a["repeats"] != 20 or a["warmups"] != 3:
        raise ValueError("formal timing requires 3 warmups and 20 repeats")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("musa", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    envs = [json.loads((p / "environment.json").read_text()) for p in (args.cuda, args.musa)]
    validate_pair(*envs)
    if [e["backend"] for e in envs] != ["cuda", "musa"]:
        raise ValueError("expected CUDA then MUSA")
    def rows(p):
        values = json.loads((p / "summary.json").read_text())
        values = [r for r in values if r["op"] == "quant_optimized" and r["scope"] == "resident_gpu"]
        if len(values) != 6 or any(r["repeats"] != 20 for r in values):
            raise ValueError("expected six formal quantization rows")
        result = {(r["format"], r["elements"]): r for r in values}
        expected = {(fmt, n) for fmt in ("mxfp8", "nvfp4") for n in (1 << 20, 4 << 20, 16 << 20)}
        if set(result) != expected:
            raise ValueError("missing, duplicate or unexpected format/size rows")
        return result
    cuda, musa = rows(args.cuda), rows(args.musa)
    if set(cuda) != set(musa):
        raise ValueError("format/size rows differ")
    lines = ["# 当前 CUDA / MUSA 固定输入对比", "",
             "本次重新测量当前保留实现，不称为历史最优。三个尺寸的实际 FP32 输入文件 SHA256 完全相同。",
             "", "条件：fixed-fp32-v1，1×N，正态输入 seed=20260909；block/nearest，MXFP8 block=32、NVFP4 block=16；预热3次、测量20次。",
             "指标：quant_optimized / resident_gpu，单位ms；NVFP4 包含全局两级归约。表中倍数是跨平台耗时比，不属于优化轮次的加速链。",
             "", "| 格式 | 元素数 | CUDA median | CUDA P95 | MUSA median | MUSA P95 | MUSA/CUDA 耗时比 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(cuda):
        c, m = cuda[key], musa[key]
        lines.append(f"| {key[0].upper()} | {key[1]//1048576}M | {c['median_ms']:.6f} | {c['p95_ms']:.6f} | "
                     f"{m['median_ms']:.6f} | {m['p95_ms']:.6f} | {m['median_ms']/c['median_ms']:.2f}x |")
    lines += ["", "设备、编译器和驱动见各次 environment.json；硬件不同，不能将差异单独归因于 GPU 算力。未锁定功耗/频率，结果代表本轮运行状态，非硬件极限。",
              "", "输入哈希：", ""]
    lines += [f"- {n} 元素：`{digest}`" for n, digest in envs[0]["input"]["sha256"].items()]
    links = [Path(os.path.relpath(p.resolve() / "RESULTS.md", args.output.resolve().parent)).as_posix()
             for p in (args.cuda, args.musa)]
    lines += ["", f"来源：[CUDA]({links[0]})、[MUSA]({links[1]})。环境文件保留日期、编译器、源码和二进制哈希。",
              "", "历史六次实验未保存实际输入哈希，不强行套入本次严格对比。今后使用同一冻结输入目录开展新轮次；各平台的第1轮基线和上一轮只在同平台内部计算。"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
