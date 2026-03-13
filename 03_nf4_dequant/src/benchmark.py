import torch
import numpy as np
import random
from get_param import parse_params

CONFIG_FILE = "./data/param.txt"
PY_FILE = "./data/py_output.bin"
CPP_FILE = "./data/cpp_output.bin"
PY_LOG = "./data/log/log_py.txt"
CPP_LOG = "./data/log/log_cpp.txt"

def load_bf16_tensor(file_path, rows, cols):
    """
    读取二进制文件，并根据原始维度进行裁剪和重塑
    """
    raw_data = np.fromfile(file_path, dtype=np.int16)
    total_original = rows * cols
    
    # 裁剪掉末尾的 Padding 元素
    if raw_data.size > total_original:
        raw_data = raw_data[:total_original]
    elif raw_data.size < total_original:
        print(f"错误: 文件数据量 ({raw_data.size}) 小于原始维度需求 ({total_original})")
        return None
    
    tensor_bf16 = torch.from_numpy(raw_data).view(torch.bfloat16)
    return tensor_bf16.reshape(rows, cols).to(torch.float32)

def main():
    config = parse_params(CONFIG_FILE)
    ROWS = config["rows"]
    COLS = config["cols"]
    
    print(f"测试维度: {ROWS} x {COLS}")
    print(f"Python: {PY_FILE}")
    print(f"C++: {CPP_FILE}")

    t_py = load_bf16_tensor(PY_FILE, ROWS, COLS)
    t_cpp = load_bf16_tensor(CPP_FILE, ROWS, COLS)

    diff = t_py - t_cpp
    abs_diff = torch.abs(diff)
    mae = torch.mean(abs_diff).item()
    mse = torch.mean(diff ** 2).item()
    max_diff = torch.max(abs_diff).item()

    exact_match_count = torch.sum(t_py == t_cpp).item()
    total_elements = t_py.numel()
    match_rate = (exact_match_count / total_elements) * 100

    print("\n" + "="*50)
    print("误差分析报告")
    print("="*50)
    print(f"平均绝对误差 (MAE):{mae:.10e}")
    print(f"均方误差 (MSE):{mse:.10e}")
    print(f"最大误差 (Max Diff):{max_diff:.10e}")
    print("-" * 50)
    print(f"完全一致元素数:{exact_match_count} / {total_elements}")
    print(f"一致率:{match_rate:.4f}%")
    print("="*50)

    print("\n 随机数据采样对比")
    print("-" * 75)
    print(f"{'Index':<10} | {'Python (BF16)':<18} | {'C++ (BF16)':<18} | {'Diff':<15}")
    print("-" * 75)

    flat_py = t_py.flatten()
    flat_cpp = t_cpp.flatten()
    
    random_indices = random.sample(range(total_elements), 10)
    
    check_indices = [0, 1, 2] + random_indices
    for idx in check_indices:
        val_py = flat_py[idx].item()
        val_cpp = flat_cpp[idx].item()
        val_diff = abs(val_py - val_cpp)
        
        print(f"{idx:<10} | {val_py:<18.6f} | {val_cpp:<18.6f} | {val_diff:<15.4e}")
    print("-" * 75)


    with open(PY_LOG, 'r', encoding='utf-8') as f:
        line = f.readline()
        py_time, py_bandwidth = [float(x.strip()) for x in line.split(',')]
        print(f"bnb耗时:{py_time:.5f}ms, 带宽:{py_bandwidth:.5f}GB/s")
    
    with open(CPP_LOG, 'r', encoding='utf-8') as f:
        line = f.readline()
        cpp_time, cpp_bandwidth = [float(x.strip()) for x in line.split(',')]
        print(f"nf4 kernel耗时:{cpp_time:.5f}ms, 带宽:{cpp_bandwidth:.5f}GB/s")
    
    

if __name__ == "__main__":
    main()