# Bilateral-Filter


## 介绍
InifiniTensor 2025冬训练营项目阶段, CUDA方向题目八: 基于CUDA的实时图像双边滤波


## 使用说明
### step1: 环境初始化
在开始编译前需配置本地openCV环境（优先使用系统自带openCV, 若缺失则尝试基于conda建立虚拟环境tmp_env并安装openCV; 若conda也不可用则项目将在缺少openCV对照的情况下直接运行）
```bash
make init PLATFORM=[nvidia, moore, metax, iluvatar]
```

### step2: 配置参数文件
修改`tester/config.txt`以配置radius, sigma_spatial和sigma_color（当radius设置为 <= 0时，代码将自适应选取滤波半径）;  
在`tester/gray/4K`和`tester/rgb/4K`路径下存放待测试的4K(3840 * 2160)图片的bin文件，分别包含10张灰白壁纸和15张RGB壁纸（具体jpeg图片参见resource下的相应路径）;  
你也可以在`tester/gray/4K`和`tester/rgb/4K`中自行添加待测试.bin文件

### step3: 编译运行
输出.bin文件和性能日志(runtime.log)将自动保存至`result`路径下
```bash
make run PLATFORM=[nvidia, moore, metax, iluvatar]
```


## 其他
清除编译和运行结果
```bash
make clean
```

清除所有生成文件(包括虚拟环境和辅助脚本)
```bash
make cleanup
```

基于各平台的结果和性能分析参见`report`路径
