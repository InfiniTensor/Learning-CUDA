"""
OpenCV 双边滤波对比验证脚本

用法:
  python3 scripts/compare_opencv.py <输入图像> <输出图像> [参数文件]

示例:
  python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v4_stream.png
  python3 scripts/compare_opencv.py test_images/yosemite.jpg output/images/yosemite_v4_stream.png params.txt

功能:
  计算 MAE 和 PSNR
"""

import cv2
import numpy as np
import sys
import os


def load_params(filepath):
    """从 params.txt 解析滤波参数，格式与 C++ 端一致"""
    params = {"radius": 5, "sigma_spatial": 3.0, "sigma_color": 30.0}

    if not os.path.exists(filepath):
        print(f"警告: 未找到参数文件 {filepath}，使用默认参数")
        return params

    with open(filepath, "r") as f:
        for line in f:
            # 去掉注释
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "radius":
                params["radius"] = int(value)
            elif key == "sigma_spatial":
                params["sigma_spatial"] = float(value)
            elif key == "sigma_color":
                params["sigma_color"] = float(value)

    return params


def compute_mae(img1, img2):
    """计算平均绝对误差"""
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))


def compute_psnr(img1, img2):
    """计算峰值信噪比"""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def main():
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <输入图像> <输出图像> [参数文件]")
        print(f"示例: {sys.argv[0]} test_images/yosemite.jpg output/images/yosemite_v2_smem.png")
        sys.exit(1)

    input_path = sys.argv[1]
    your_output_path = sys.argv[2]
    param_path = sys.argv[3] if len(sys.argv) > 3 else "params.txt"

    # 读取参数
    params = load_params(param_path)
    radius = params["radius"]
    sigma_color = params["sigma_color"]
    sigma_spatial = params["sigma_spatial"]

    print(f"滤波参数: radius={radius}, sigma_spatial={sigma_spatial}, sigma_color={sigma_color}")

    # 读取图像
    input_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    your_output = cv2.imread(your_output_path, cv2.IMREAD_UNCHANGED)

    if input_img is None:
        print(f"错误: 无法读取输入图像 {input_path}")
        sys.exit(1)
    if your_output is None:
        print(f"错误: 无法读取输出图像 {your_output_path}")
        sys.exit(1)

    # OpenCV bilateralFilter
    # 参数: d = 窗口直径 (2*radius+1), sigmaColor, sigmaSpace
    d = 2 * radius + 1
    opencv_output = cv2.bilateralFilter(input_img, d, sigma_color, sigma_spatial)

    # 保存 OpenCV 结果到 output/images/
    os.makedirs("output/images", exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    opencv_out_path = f"output/images/{basename}_opencv.png"
    cv2.imwrite(opencv_out_path, opencv_output)
    print(f"OpenCV 结果已保存: {opencv_out_path}")

    # 计算该阶段版本结果 vs OpenCV
    mae = compute_mae(your_output, opencv_output)
    psnr = compute_psnr(your_output, opencv_output)

    print(f"\n===== 该阶段结果 vs OpenCV =====")
    print(f"  MAE:  {mae:.4f}  {'[通过]' if mae < 1.0 else '[未通过: MAE >= 1]'}")
    print(f"  PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()
