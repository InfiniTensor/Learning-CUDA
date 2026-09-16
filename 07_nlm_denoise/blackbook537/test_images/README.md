# 示例图像来源

图片由本项目生成器产生，没有采用参考作业的图片或输出。所有文件为 8 位 RGB PNG。

| 文件 | 用途/来源 |
|---|---|
| `clean_1920x1080_3ch.png` | 平滑渐变参考图，生成器噪声幅度 0 |
| `noisy_1920x1080_3ch_sigma25.png` | 对应 1080p 含噪图，seed=7、均匀噪声幅度 25 |
| `noisy_256x256_3ch_sigma25.png` | 默认快速运行输入；历史文件，不用于正式质量表 |
| `denoised_1920x1080_base_v2.png` | 从历史 MetaX 证据中保留的 V2/base 输出，仅作演示 |

生成器使用 LCG 确定性序列，将均匀噪声加到渐变图后量化、裁剪到 0–255。文件名中的 `sigma25` 是旧命名，含义为均匀噪声幅度，并非高斯标准差 25。灰度、复杂纹理、自然图像与标准差受控噪声的补充建议见 [SUBMISSION.md](../SUBMISSION.md)。

复现 1080p 配对图（输出到被忽略的运行目录）：

```bash
./build/nlm_denoise gen --size 1920x1080 --channels 3 --sigma 0 --seed 7 -o output/work/clean.png
./build/nlm_denoise gen --size 1920x1080 --channels 3 --sigma 25 --seed 7 -o output/work/noisy.png
./build/nlm_denoise run -i output/work/noisy.png -o output/images/denoised.png -p params.txt --kernel 2
```

`SHA256SUMS.txt` 记录当前四张图的哈希。可在本目录执行 `sha256sum -c SHA256SUMS.txt`。
