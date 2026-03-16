## NF4 反量化
author: flashzxi

本项目是利用cuda高效计算nf4反量化，对比bitsandbytes 实现

本项目的假设：
每个block大小为64个元素

二级量化每个group包含256个block.

## 实现
总共实现了三个版本，一个最简单的naive版本，一个二级反量化和一级反量化分开计算的版本以及最终的单独kernel解两层反量化的版本。其中naive版本在`src/nf4_dequant_naive.cu`,其余两个版本都在`src/nf4_dequant_warp8.cu`



开发工程中，我尝试