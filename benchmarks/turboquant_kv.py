"""Compare experimental KV codecs on CPU without claiming A55 throughput."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "finetune" / "modeling")]

import common
from turboquant_kv import TurboQuantKVCodec, tensor_bytes


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distribution", choices=("normal", "outliers"), default="normal")
    parser.add_argument("--capture",type=Path,
                        help="NPZ containing standardized queries, keys, and values")
    parser.add_argument("--out", help="optional JSON result path")
    return parser.parse_args()


def timed(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def nmse(reference, actual):
    return float((reference - actual).square().mean() /
                 reference.square().mean().clamp_min(1e-12))


def cosine(reference, actual):
    numerator = (reference * actual).sum(-1)
    denominator = reference.norm(dim=-1) * actual.norm(dim=-1)
    return float((numerator / denominator.clamp_min(1e-12)).mean())


def correlation(reference, actual):
    x = reference.flatten().double(); y = actual.flatten().double()
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))


def topk_overlap(reference, actual, k):
    exact = reference.topk(k, dim=-1).indices
    approximate = actual.topk(k, dim=-1).indices
    overlap = (exact.unsqueeze(-1) == approximate.unsqueeze(-2)).any(-1).float().mean()
    return float(overlap)


def quality(keys, values, reconstructed_keys, reconstructed_values,
            exact_scores, scores, exact_output, output):
    return {
        "key_nmse": nmse(keys, reconstructed_keys),
        "key_cosine": cosine(keys, reconstructed_keys),
        "value_nmse": nmse(values, reconstructed_values),
        "value_cosine": cosine(values, reconstructed_values),
        "score_rmse": float((exact_scores - scores).square().mean().sqrt()),
        "score_correlation": correlation(exact_scores, scores),
        "top8_overlap": topk_overlap(exact_scores, scores, min(8, scores.shape[-1])),
        "output_nmse": nmse(exact_output, output),
        "output_cosine": cosine(exact_output, output),
    }


def finalize(queries, keys, values, reconstructed_keys, reconstructed_values, scores):
    width = keys.shape[-1]
    exact_scores = torch.einsum("hqd,htd->hqt", queries, keys) / math.sqrt(width)
    exact_output = exact_scores.softmax(-1) @ values
    output = scores.softmax(-1) @ reconstructed_values
    return quality(keys, values, reconstructed_keys, reconstructed_values,
                   exact_scores, scores, exact_output, output)


def evaluate_1bit(queries, keys, values, repeats):
    heads, tokens, width = keys.shape
    codec_k = common.KVCodec1(heads, width, seed=101).eval()
    codec_v = common.KVCodec1(heads, width, seed=211).eval()
    batch_k = keys.unsqueeze(0); batch_v = values.unsqueeze(0)
    codec_k.calibrate(batch_k); codec_v.calibrate(batch_v)
    encode = lambda: (codec_k.pack(batch_k), codec_v.pack(batch_v))
    packed_k, packed_v = encode()

    def decode_and_score():
        rk = codec_k.unpack(packed_k)[0]; rv = codec_v.unpack(packed_v)[0]
        scores = torch.einsum("hqd,htd->hqt", queries, rk) / math.sqrt(width)
        return rk, rv, scores

    rk, rv, scores = decode_and_score()
    result = finalize(queries, keys, values, rk, rv, scores)
    result.update({
        "payload_bytes_per_token_head":
            (packed_k.numel() + packed_v.numel()) / (heads * tokens),
        "encode_ms": timed(encode, repeats),
        "decode_and_score_ms": timed(decode_and_score, repeats),
    })
    return result


def evaluate_2bit(queries, keys, values, repeats):
    heads, tokens, width = keys.shape
    encode = lambda: (*common.kv2_pack(keys), *common.kv2_pack(values))
    packed_k, scale_k, packed_v, scale_v = encode()

    def decode_and_score():
        rk = common.kv2_unpack(packed_k, scale_k)
        rv = common.kv2_unpack(packed_v, scale_v)
        scores = torch.einsum("hqd,htd->hqt", queries, rk) / math.sqrt(width)
        return rk, rv, scores

    rk, rv, scores = decode_and_score()
    result = finalize(queries, keys, values, rk, rv, scores)
    result.update({
        "payload_bytes_per_token_head":
            tensor_bytes((packed_k, scale_k, packed_v, scale_v)) / (heads * tokens),
        "encode_ms": timed(encode, repeats),
        "decode_and_score_ms": timed(decode_and_score, repeats),
    })
    return result


def evaluate_turboquant(queries, keys, values, repeats):
    heads, tokens, width = keys.shape
    codec = TurboQuantKVCodec(width, key_bits=3, value_bits=4, group_size=32).eval()
    encode = lambda: (codec.keys.pack(keys), codec.values.pack(values))
    packed_k, packed_v = encode()

    def decode_and_score():
        rk = codec.keys.unpack(packed_k); rv = codec.values.unpack(packed_v)
        scores = codec.keys.scores(queries, packed_k) / math.sqrt(width)
        return rk, rv, scores

    rk, rv, scores = decode_and_score()
    result = finalize(queries, keys, values, rk, rv, scores)
    result.update({
        "payload_bytes_per_token_head":
            (tensor_bytes(packed_k) + tensor_bytes(packed_v)) / (heads * tokens),
        "encode_ms": timed(encode, repeats),
        "decode_and_score_ms": timed(decode_and_score, repeats),
    })
    return result


def evaluate_int4(queries,keys,values,repeats):
    heads,tokens,width=keys.shape
    encode=lambda:(*common.kv4_key_pack(keys),*common.kv4_value_pack(values))
    packed_k,scale_k,packed_v,scale_v,minimum_v=encode()

    def decode_and_score():
        rk=common.kv4_key_unpack(packed_k,scale_k,torch.float32)
        rv=common.kv4_value_unpack(packed_v,scale_v,minimum_v,dtype=torch.float32)
        # PyTorch reference; the native A55 path should score packed signed
        # nibbles using replicated INT8 query codes and SDOT.
        scores=torch.einsum("hqd,htd->hqt",queries,rk)/math.sqrt(width)
        return rk,rv,scores

    rk,rv,scores=decode_and_score()
    result=finalize(queries,keys,values,rk,rv,scores)
    result.update({
        "payload_bytes_per_token_head":tensor_bytes(
            (packed_k,scale_k,packed_v,scale_v,minimum_v))/(heads*tokens),
        "encode_ms":timed(encode,repeats),
        "decode_and_score_ms":timed(decode_and_score,repeats),
    })
    return result


def main():
    args = arguments()
    if min(args.tokens, args.queries, args.heads, args.repeats) < 1:
        raise SystemExit("tokens, queries, heads, and repeats must be positive")
    if args.capture:
        import numpy as np
        capture=np.load(args.capture)
        queries=torch.from_numpy(capture["queries"]).float()
        keys=torch.from_numpy(capture["keys"]).float()
        values=torch.from_numpy(capture["values"]).float()
        if keys.shape!=values.shape or queries.ndim!=3 or keys.ndim!=3:
            raise SystemExit("capture must contain HxQx64 queries and matching HxTx64 K/V")
        if queries.shape[0]!=keys.shape[0] or queries.shape[-1]!=64 or keys.shape[-1]!=64:
            raise SystemExit("capture query/K/V head counts and 64-wide dimensions must match")
    else:
        torch.manual_seed(args.seed)
        keys = torch.randn(args.heads, args.tokens, 64)
        values = torch.randn_like(keys)
        queries = torch.randn(args.heads, args.queries, 64)
        if args.distribution == "outliers":
            keys[..., ::17] *= 12
            values[..., ::19] *= 12
            queries[..., ::17] *= 12
    result = {
        "host": "CPU timing; not an A55 throughput claim",
        "shape": {"heads": keys.shape[0], "tokens": keys.shape[1],
                  "queries": queries.shape[1], "head_dim": 64},
        "distribution": "captured" if args.capture else args.distribution,
        "codecs": {
            "shadow_1bit": evaluate_1bit(queries, keys, values, args.repeats),
            "shadow_2bit": evaluate_2bit(queries, keys, values, args.repeats),
            "a55_int4_kv": evaluate_int4(queries,keys,values,args.repeats),
            "turboquant_k3_v4": evaluate_turboquant(queries, keys, values, args.repeats),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
