"""Shared Arm integer-FFN QAT setup and diagnostics."""

import torch

import common


def configure(enabled=True,weight_dtype="ternary",activation_strength=1.0):
    """Configure deployment-exact FFN activation boundaries globally."""
    common.set_ffn_qat(weight_dtype,enabled,activation_strength)
    return contract(enabled,weight_dtype)


def contract(enabled=True,weight_dtype="ternary"):
    if weight_dtype not in {"ternary","int4_row"}:
        raise ValueError(f"unsupported FFN weight dtype {weight_dtype!r}")
    return {
        "ffn_act_qat": bool(enabled),
        "ffn_act_bits": 8,
        "ffn_act_scale": "per_token_power_of_two",
        "ffn_weight_dtype": weight_dtype,
        "ffn_weight_scale": "per_output_row_fp32",
        "ffn_accumulator": "int32",
    }


def activation_qat_strength(consumed_tokens,warmup_tokens,enabled=True):
    if not enabled: return 0.0
    if warmup_tokens<=0: return 1.0
    return min(1.0,max(0.0,float(consumed_tokens)/float(warmup_tokens)))


def checkpoint_contract(cfg):
    """Extract fields that must match when a training run resumes."""
    return {key:cfg.get(key) for key in ("ffn_weight_dtype","ffn_act_qat",
        "ffn_act_bits","ffn_act_scale","ffn_weight_scale","ffn_accumulator")}


@torch.no_grad()
def activation_stats(values):
    """Summarize one FFN boundary using the exact deployment quantizer."""
    code, scale, dequantized = common.int8_pot_values(values)
    error = (values.float() - dequantized.float()).square()
    signal = values.float().square()
    return {
        "saturation_fraction": float((code.abs() == 127).float().mean()),
        "zero_fraction": float((code == 0).float().mean()),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "nmse": float(error.sum() / signal.sum().clamp_min(1e-30)),
    }


def autocast_and_scaler(device, amp_dtype):
    """Return autocast settings and loss scaler without changing FP32 parameters."""
    if amp_dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported AMP dtype {amp_dtype!r}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[amp_dtype]
    enabled = device.type == "cuda" and amp_dtype != "fp32"
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and amp_dtype=="fp16")
    return dtype, enabled, scaler
