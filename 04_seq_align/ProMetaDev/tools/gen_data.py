#!/usr/bin/env python3
"""生成合成参考基因组（FASTA）+ 测序 reads（FASTQ）+ 真值文件。

用法示例:
  python3 gen_data.py --ref-seqs 5 --ref-len 20000 --reads 200 --read-len 150 \
      --sub-rate 0.03 --random-frac 0.2 --seed 1 \
      --out-ref ref.fa --out-reads reads.fq --out-truth truth.txt

真值文件每行: <seq_id> <pos> <score>
  - 来自参考的 read：seq_id/pos 为采样起点，score 为无空位情况下的理论得分
  - 随机 read：-1 -1 0（期望被判定为 unknown_origin）
"""
import argparse
import random

BASES = "ACGT"

# 把任意字节映射到 A/C/G/T 的查表（用于快速生成大序列）
_BYTE_TO_BASE = bytes([ord(BASES[b & 3]) for b in range(256)])


def rand_seq(rng, n):
    """生成 n 个随机碱基。大 n 时用 randbytes + 查表，比逐字符快两个数量级。"""
    try:
        return rng.randbytes(n).translate(_BYTE_TO_BASE).decode("ascii")
    except AttributeError:                      # Python < 3.9
        return "".join(rng.choice(BASES) for _ in range(n))


def mutate_sub(rng, s, rate):
    """只做替换突变，返回 (新序列, 替换数)"""
    out = list(s)
    n_sub = 0
    for i, c in enumerate(out):
        if rng.random() < rate:
            choices = [b for b in BASES if b != c]
            out[i] = rng.choice(choices)
            n_sub += 1
    return "".join(out), n_sub


def mutate_indel(rng, s, rate):
    """在替换的基础上加入插入/删除，返回 (新序列, 净变化)"""
    out = list(s)
    i = 0
    while i < len(out):
        r = rng.random()
        if r < rate / 2:          # 删除
            del out[i]
            continue
        if r < rate:              # 插入
            out.insert(i, rng.choice(BASES))
            i += 2
            continue
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-seqs", type=int, default=5)
    ap.add_argument("--ref-len", type=int, default=20000)
    ap.add_argument("--reads", type=int, default=200)
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--sub-rate", type=float, default=0.03, help="替换突变率")
    ap.add_argument("--indel-rate", type=float, default=0.0, help="插入/删除率（0=不加）")
    ap.add_argument("--random-frac", type=float, default=0.2, help="随机 read 占比")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-ref", required=True)
    ap.add_argument("--out-reads", required=True)
    ap.add_argument("--out-truth", required=True)
    ap.add_argument("--line-width", type=int, default=70, help="FASTA 每行碱基数")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- 参考基因组 ----
    seqs = [rand_seq(rng, args.ref_len) for _ in range(args.ref_seqs)]
    w = args.line_width
    with open(args.out_ref, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">chr{i+1}\n")
            f.write("\n".join(s[j:j + w] for j in range(0, len(s), w)))
            f.write("\n")

    # ---- reads ----
    reads = []
    truth = []
    n_random = int(round(args.reads * args.random_frac))
    n_from_ref = args.reads - n_random

    for i in range(args.reads):
        name = f"read{i}"
        if i < n_from_ref:
            sid = rng.randrange(args.ref_seqs)
            pos = rng.randrange(0, args.ref_len - args.read_len)
            frag = seqs[sid][pos:pos + args.read_len]
            if args.indel_rate > 0:
                seq = mutate_indel(rng, frag, args.indel_rate)
                seq, _ = mutate_sub(rng, seq, args.sub_rate)
                score = -1                      # 含空位时真值得分不解析给出
            else:
                seq, n_sub = mutate_sub(rng, frag, args.sub_rate)
                # 无空位：score = (L-n_sub)*2 + n_sub*(-1)
                score = (args.read_len - n_sub) * 2 - n_sub
            seq = seq[:args.read_len]           # 长度对齐
            truth.append((sid, pos, score))
        else:
            seq = rand_seq(rng, args.read_len)
            truth.append((-1, -1, 0))
        reads.append((name, seq))

    with open(args.out_reads, "w") as f:
        for name, seq in reads:
            f.write(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n")

    with open(args.out_truth, "w") as f:
        for sid, pos, score in truth:
            f.write(f"{sid} {pos} {score}\n")

    total = args.ref_seqs * args.ref_len
    print(f"[gen] 参考: {args.ref_seqs} 条 × {args.ref_len} bp = {total} bp -> {args.out_ref}")
    print(f"[gen] reads: {args.reads} 条 × {args.read_len} bp "
          f"(来自参考 {n_from_ref}, 随机 {n_random}) -> {args.out_reads}")


if __name__ == "__main__":
    main()
