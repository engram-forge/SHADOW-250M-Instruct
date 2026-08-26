"""Export a fine-tuned checkpoint to the 52 MB deploy file for the CPU runtime.
    python export_model.py my_model/finetuned.pt my_model.shdw
"""
import json, os, sys, pathlib
for k, v in {"SHADOW_D": "1536", "SHADOW_NL": "10", "SHADOW_NH": "24", "SHADOW_NKV": "2", "SHADOW_HD": "64",
             "SHADOW_FFNH": "4224", "SHADOW_FAST_ATTN": "1", "SHADOW_KV_BITS": "1", "SHADOW_KV_TWO_TIER": "1"}.items():
    os.environ.setdefault(k, v)
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "modeling"))
import subprocess
import torch
src, dst = sys.argv[1], sys.argv[2]
checkpoint = torch.load(src, map_location="cpu", weights_only=False)
cfg = checkpoint.get("cfg", {})
weight_dtype=cfg.get("ffn_weight_dtype","ternary")
if weight_dtype not in {"ternary","int4_row"}: raise SystemExit(f"unsupported FFN dtype {weight_dtype}")
mtp_horizon=int(cfg.get("mtp_horizon",1))
if mtp_horizon not in (1,2): raise SystemExit(f"invalid checkpoint MTP horizon {mtp_horizon}")
manifest = {
    "version": 1,
    "target": "armv8.2-a-dotprod-cortex-a55",
    "architecture": "swiglu",
    "mtp": {"horizon":mtp_horizon,
            "auxiliary_heads":mtp_horizon-1,
            "variant":"a55_k2_conditioned_residual_mlp" if mtp_horizon==2 else "none",
            "deepseek_exact":False,
            "module_type":"token_conditioned_residual_mlp" if mtp_horizon==2 else "none",
            "hidden_width":int(os.environ["SHADOW_D"])//2 if mtp_horizon==2 else 0,
            "vocabulary_projection":"shared_base_head_fingerprint_table_and_tied_bias",
            "training_loss_weight":float(cfg.get("mtp_loss_weight",0.0))},
    "ffn_weight": {"dtype":weight_dtype,"scale":"per_output_row_fp32",
                   "disk_layout":"base3_5trits_per_byte" if weight_dtype=="ternary"
                                 else "twos_complement_int4_2_per_byte"},
    "ffn_activation": {
        "enabled": bool(cfg.get("ffn_act_qat", False)),
        "dtype": "int8" if cfg.get("ffn_act_qat", False) else "float32",
        "scale": cfg.get("ffn_act_scale", "none"),
        "boundaries": ["post_ffn_rmsnorm_shared_up_gate", "post_swiglu_pre_down"],
    },
    "accumulator": "int32" if cfg.get("ffn_act_qat", False) else "float32",
    "required_cpu_features": ["asimd", "asimddp"],
    "runtime_dispatch": {"linux_hwcap": "HWCAP_ASIMDDP",
                         "fallback": "armv8_a_widening_or_portable"},
    "execution_layout_candidates": (["biased_ternary_nibble_sdot","signed_int8_trit_sdot"]
        if weight_dtype=="ternary" else ["twos_complement_int4_nibble_sdot","signed_int8_sdot"]),
    "compatible_with_bundled_engine": (weight_dtype=="ternary" and
                                        not bool(cfg.get("ffn_act_qat",False)) and
                                        mtp_horizon==1),
}
tmp = dst + ".full"
try:
    r = subprocess.run([sys.executable, str(HERE / "modeling" / "export_ternary.py"), src, tmp],
                       env={**os.environ, "PYTHONPATH": str(HERE / "modeling")})
    if r.returncode: sys.exit(r.returncode)
    r = subprocess.run([sys.executable, str(HERE / "modeling" / "repack_shdw.py"), tmp, dst, "--fp16"])
    if r.returncode: sys.exit(r.returncode)
finally:
    if os.path.exists(tmp): os.remove(tmp)
pathlib.Path(dst + ".a55.json").write_text(json.dumps(manifest, indent=2) + "\n")
if not manifest["compatible_with_bundled_engine"]:
    print("warning: this checkpoint requires the planned A55 DotProd FFN engine; "
          "the bundled ternary/FP32-activation binary is not deployment-exact")
print("wrote", dst)
print("wrote", dst + ".a55.json")
