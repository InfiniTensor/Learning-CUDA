import torch
from cuda import nvfp4_cuda, mxfp8_cuda

def quant_dequant(vals, rows, cols, format, scale_mode, block_size, output_type):
    data = torch.tensor(vals, dtype=torch.float32, device="cuda")

    fmt = nvfp4_cuda if format == "nvfp4" else mxfp8_cuda
    scale_mode = 1 if scale_mode == "tensor" else 0

    start = torch.cuda.Event(enable_timing=True)
    mid = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    packed, scales = fmt.quant(data, rows, cols, block_size, scale_mode)
    mid.record()
    dequant = fmt.dequant(packed, scales, rows, cols, block_size, output_type)
    end.record()
    end.synchronize()
    t_quant = start.elapsed_time(mid) / 1000.0     
    t_dequant = mid.elapsed_time(end) / 1000.0 

    return packed.cpu().numpy().tobytes(), scales.cpu().tolist(), dequant.cpu().tolist(), t_quant, t_dequant