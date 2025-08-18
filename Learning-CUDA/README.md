# Learning-CUDA

本项目为 2025 年夏季 InfiniTensor 大模型与人工智能系统训练营 CUDA 方向专业阶段的作业系统。

## 📁 项目结构

```text
learning-CUDA/
├── Makefile
├── README
├── src
│   └── kernels.cu
└── tester
    ├── tester.o
    └── utils.h
```  

## 环境配置

如果你使用的是训练营提供的服务器，则该步骤可直接跳过。

请确保系统已安装以下工具：

1. **CUDA Toolkit**（版本11.0及以上）：
    - 验证安装：运行`nvcc --version`。
    - 安装：从[NVIDIA CUDA Toolkit下载页](https://developer.nvidia.com/cuda-downloads)获取。
2. **GNU Make**：
    - 验证安装：运行`make --version`（大多数Linux/macOS已预装）。

## 🧠 作业

作业一共有两题。需实现 `src/kernels.cu` 中给定的 **2 个 CUDA  函数** 。

1. **kthLargest**

实现 CUDA 的 kthLargest 函数。给定一个连续的输入数组和非负数 k，返回该数组中第 k 大的数。该函数需支持 int 和 float 两种类型的输入。具体边界处理和一些条件可见文件中的注释。
  
2. **flashAttention**

实现 Flash Attention 算子。需支持 causal masking 和 GQA。具体行为与 [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 保持一致。接口未提供的参数所代表的功能无需支持和实现。具体参数要求请参考文件中的注释。该函数需支持 float。

### 注意事项

1. **禁止抄袭与舞弊**，包括抄袭其他学员的代码和开源实现。可以讨论和参考思路，但禁止直接看/抄代码。一经发现，成绩作废并失去进入项目阶段和后续实习与推荐等资格；
2. 两个题目都**禁止使用任何库函数**来直接实现关键功能；
3. 主要计算均需在 GPU 上实现；如有一些信息和程序准备性质的（例如元信息计算/转换、资源准备等）则可以在 CPU/Host 上进行；
4. 代码风格不限，但需保持一致；
5. 需进行**适当**的代码注释解释重要部分；

### 提交方式
在网站 [InfiniTensor 开源社区](https://beta.infinitensor.com/camp/summer2025) 上提交 GitHub 链接，以最新提交为准。

## 🛠️ 编译与运行

  代码编译与运行可以使用提供的 `Makefile` 十分简便的实现。

### 构建与运行指令

使用`Makefile`简化构建流程，以下命令需在**项目根目录**（即 `Makefile` 所在的目录）执行：

#### 默认：构建并运行测试（非 verbose 模式）

直接在命令行使用 `make` 指令编译代码并执行测试，输出简洁结果。

#### 构建并运行测试（verbose 模式）

直接在命令行使用 `make VERBOSE=true` 指令编译代码并执行测试，输出包括执行时间在内的结果。

## 📊 评分规则

本次作业的评分标准如下：

1. **正确性优先**  
   - 所有提交首先以正确性为前提，需在提供的测试用例中正确输出结果；
   - 正确性提供基础分：每通过一个测例，获得相应的基础得分；
   - 未通过的测试用例，不计入性能排名；
   - 不符合**注意事项**中要求的，不得分。

2. **性能加分**  
   - 在正确性的基础上，会对各实现的性能进行排名；
   - 性能越优，获得的额外分数越多；
   - 性能评判将在提供的服务器上进行，因此请在服务器上进行性能评估。

3. **最终成绩**  
   - 总体得分由「通过的测试用例数量」与「性能排名加分」共同决定。  
   - 各测试用例的分数相加，形成最终成绩。  

## 📬 有疑问?

可以在群里直接询问助教！

Good luck and happy coding! 🚀
