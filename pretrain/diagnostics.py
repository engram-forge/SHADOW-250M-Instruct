"""Low-frequency, compact stability diagnostics for quantization-aware training."""

import time

import torch


@torch.no_grad()
def normalized_participation(values):
    """Return effective-coordinate count divided by tensor size, in [1/n, 1]."""
    values = values.detach().float()
    if not values.numel():
        return 0.0
    maximum = values.abs().amax()
    if not bool(torch.isfinite(maximum)) or float(maximum) == 0.0:
        return 0.0
    scaled_square = (values / maximum).square()
    mean_square = scaled_square.mean()
    mean_fourth = scaled_square.square().mean()
    return float((mean_square.square() / mean_fourth.clamp_min(1e-300)).clamp(0, 1))


@torch.no_grad()
def gradient_participation(model, minimum_tensor_size=256):
    """Summarize global and per-parameter gradient concentration."""
    gradients = [
        (name, parameter.grad.detach())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"normalized_global": 0.0, "layer_median": 0.0,
                "layer_p10": 0.0, "worst_name": None, "worst": 0.0}

    global_max = torch.stack([gradient.abs().amax().float() for _, gradient in gradients]).amax()
    if not bool(torch.isfinite(global_max)) or float(global_max) == 0.0:
        global_ratio = 0.0
    else:
        sum_square = torch.zeros((), dtype=torch.float32, device=global_max.device)
        sum_fourth = torch.zeros_like(sum_square)
        count = 0
        for _, gradient in gradients:
            square = (gradient.float() / global_max).square()
            sum_square += square.sum()
            sum_fourth += square.square().sum()
            count += square.numel()
        global_ratio = float(
            (sum_square.square() / (count * sum_fourth.clamp_min(1e-300))).clamp(0, 1)
        )

    layers = [
        (name, normalized_participation(gradient))
        for name, gradient in gradients if gradient.numel() >= minimum_tensor_size
    ]
    layers.sort(key=lambda item: item[1])
    if not layers:
        return {"normalized_global": global_ratio, "layer_median": 0.0,
                "layer_p10": 0.0, "worst_name": None, "worst": 0.0}
    values = torch.tensor([value for _, value in layers], dtype=torch.float64)
    return {
        "normalized_global": global_ratio,
        "layer_median": float(values.median()),
        "layer_p10": float(torch.quantile(values, 0.1)),
        "worst_name": layers[0][0],
        "worst": layers[0][1],
    }


@torch.no_grad()
def quantization_gap(model,rvq_type,weight_dtype="ternary"):
    """Return weighted NMSE and the worst module for RVQ and selected FFN weights."""
    if weight_dtype not in {"ternary","int4_row"}: raise ValueError(weight_dtype)
    families = {
        "rvq": {"error": 0.0, "signal": 0.0, "worst_name": None, "worst": -1.0},
        weight_dtype: {"error": 0.0, "signal": 0.0, "worst_name": None, "worst": -1.0},
    }
    for name, module in model.named_modules():
        if not isinstance(module, rvq_type):
            continue
        weight = module.weight.detach().float()
        if module.g == 32:
            family=weight_dtype
            if weight_dtype=="ternary":
                scale=weight.abs().mean(dim=1,keepdim=True).clamp_min(1e-5); quantized=(weight/scale).round().clamp(-1,1)*scale
            else:
                scale=(weight.abs().amax(dim=1,keepdim=True)/7).clamp_min(1e-8); quantized=(weight/scale).round().clamp(-7,7)*scale
        else:
            family = "rvq"
            if module._q is None:
                continue
            quantized = module._q.detach().float()
        error = float((weight - quantized).square().sum())
        signal = float(weight.square().sum())
        nmse = error / max(signal, 1e-30)
        stats = families[family]
        stats["error"] += error
        stats["signal"] += signal
        if nmse > stats["worst"]:
            stats["worst"] = nmse
            stats["worst_name"] = name

    return {
        family: {
            "nmse": stats["error"] / max(stats["signal"], 1e-30),
            "worst_name": stats["worst_name"],
            "worst_nmse": max(0.0, stats["worst"]),
        }
        for family, stats in families.items()
    }


def stability_diagnostics(model,rvq_type,update,tokens,gradients=None,started_at=None,weight_dtype="ternary"):
    """Build one compact diagnostic record and report measurement overhead."""
    started = time.perf_counter() if started_at is None else started_at
    record = {
        "update": update,
        "tokens": tokens,
        "gradient_participation": gradients or gradient_participation(model),
        "qat_gap": quantization_gap(model,rvq_type,weight_dtype),
    }
    record["measurement_seconds"] = time.perf_counter() - started
    return record
