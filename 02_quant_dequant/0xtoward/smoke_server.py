"""Real HTTP generation with packed weights -> our CUDA dequant -> ordinary F.linear.

This is a deliberately small Transformers integration, not a vLLM/Marlin patch.
Only text decoder Linear weights are quantized. Embedding/head/norm/vision remain BF16.
"""

import argparse
import gc
import json
from pathlib import Path
import threading
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from codec import QuantizedTensor, quantize, dequantize


class PackedLinear(nn.Module):
    def __init__(self, linear, mode):
        super().__init__()
        opts = (
            dict(format="mxfp8", scale_policy="rceil")
            if mode == "mxfp8"
            else dict(format="nvfp4", four_over_six=mode == "nvfp4_4over6")
        )
        qt = quantize(linear.weight.detach().cuda().contiguous(), **opts)
        self.register_buffer("packed", qt.data)
        self.register_buffer("scales", qt.scales)
        self.register_buffer("global_scale", qt.global_scale)
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().cuda()
        )
        self.metadata = {
            k: v
            for k, v in vars(qt).items()
            if k not in ("data", "scales", "global_scale")
        }
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        qt = QuantizedTensor(
            self.packed, self.scales, self.global_scale, **self.metadata
        )
        weight = dequantize(qt, x.dtype)
        return F.linear(x, weight, self.bias)


def load_model(path, mode):
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    cls = (
        AutoModelForImageTextToText
        if cfg.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = cls.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    eligible = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        if cfg.model_type == "qwen3_5" and "language_model.layers." not in name:
            continue
        if cfg.model_type != "qwen3_5" and ".layers." not in name:
            continue
        eligible.append((name, module))
    count = 0
    before = 0
    after = 0
    if mode != "bf16":
        for name, linear in eligible:
            parent_name, key = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            replacement = PackedLinear(linear, mode)
            setattr(parent, key, replacement)
            before += linear.weight.numel() * linear.weight.element_size()
            after += (
                replacement.packed.numel()
                + replacement.scales.numel()
                + (4 if mode != "mxfp8" else 0)
            )
            count += 1
    # Drop references to the old dense CPU parameters before loading remaining tensors.
    del eligible
    gc.collect()
    torch.cuda.empty_cache()
    model = model.cuda().eval()
    info = dict(
        model=str(path),
        mode=mode,
        quantized_linear_count=count,
        original_linear_bytes=before,
        packed_linear_bytes=after,
        allocated_after_load_bytes=torch.cuda.memory_allocated(),
        gpu=torch.cuda.get_device_name(),
        backend="Transformers eager/SDPA; manual CUDA dequant + F.linear; no Marlin/no graph",
    )
    print("READY_INFO " + json.dumps(info), flush=True)
    return model, tokenizer, info


class Request(BaseModel):
    messages: list[dict]
    model: str = "local"
    max_tokens: int = 64
    temperature: float = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--mode", choices=["bf16", "mxfp8", "nvfp4", "nvfp4_4over6"], required=True
    )
    p.add_argument("--port", type=int, default=8127)
    args = p.parse_args()
    torch.manual_seed(2026)
    torch.set_num_threads(8)
    model, tok, info = load_model(args.model, args.mode)
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return dict(status="ok", **info)

    @app.get("/v1/models")
    def models():
        return dict(data=[dict(id=Path(args.model).name, object="model")])

    @app.post("/v1/chat/completions")
    def chat(req: Request):
        with lock, torch.inference_mode():
            text = tok.apply_chat_template(
                req.messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tok(text, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            out = model.generate(
                **inputs,
                max_new_tokens=min(max(req.max_tokens, 1), 256),
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ids = out[0, inputs.input_ids.shape[1] :]
            answer = tok.decode(ids, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            finish = "stop" if ids.numel() and ids[-1].item() in eos_ids else "length"
            return dict(
                object="chat.completion",
                model=Path(args.model).name,
                choices=[
                    dict(
                        index=0,
                        message=dict(role="assistant", content=answer),
                        finish_reason=finish,
                    )
                ],
                usage=dict(
                    prompt_tokens=inputs.input_ids.numel(),
                    completion_tokens=ids.numel(),
                ),
                measurement=dict(
                    elapsed_seconds=elapsed,
                    output_tokens_per_second=ids.numel() / elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                ),
            )

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
