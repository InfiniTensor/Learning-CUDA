import re
import os

def parse_params(config_path):
    # 默认参数，防止 param.txt 缺失某些字段
    params = {
        "blocksize": "128",
        "groupsize": "128", 
        "compute_type": "bf16",
        "rows": "1024",
        "cols": "1024"
    }

    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 移除注释和首尾空格
            line = re.sub(r'#.*', '', line).strip()
            if not line: continue
            if '=' in line:
                key, value = line.split('=')
                params[key.strip()] = value.strip().strip('"').strip("'")
    
    # 将解析结果转换为正确的类型
    return {
        "blocksize": int(params.get("blocksize", 128)),
        "groupsize": int(params.get("groupsize", 128)),
        "compute_type": params.get("compute_type", "bf16").lower(),
        "rows": int(params.get("rows", 1024)),
        "cols": int(params.get("cols", 1024))
    }