"""Run the historical Linux x86 deployment on Modal and capture token goldens."""
import json
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = modal.App("shadow-linux-runtime-golden")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(ROOT / "deployment/bin/linux/shadow", "/workspace/deployment/bin/linux/shadow", copy=True)
    .add_local_file(ROOT / "deployment/shadow250m_instruct.shdw", "/workspace/deployment/shadow250m_instruct.shdw", copy=True)
    .add_local_file(ROOT / "deployment/fp131072.npy", "/workspace/deployment/fp131072.npy", copy=True)
    .add_local_dir(ROOT / "benchmarks", "/workspace/benchmarks", copy=True)
)


@app.function(image=image, cpu=4, memory=1024, timeout=10 * 60, max_containers=32)
def generate_one(case, generate_tokens=8):
    root = pathlib.Path("/workspace")
    kernel = root / "deployment/bin/linux/shadow"
    kernel.chmod(0o755)
    model = root / "deployment/shadow250m_instruct.shdw"
    table = root / "deployment/fp131072.npy"
    result = subprocess.run(
        [str(kernel), str(model), str(table), " ".join(map(str, case["tokens"])), str(generate_tokens)],
        capture_output=True, text=True, check=True, env={"SHADOW_THREADS": "4"},
    )
    return {"id": case["id"], "output": [int(value) for value in result.stdout.split()]}


@app.local_entrypoint()
def main(fixture="benchmarks/pirate_runtime_fixture.json", out="benchmarks/pirate_linux_golden.json", generate_tokens=8):
    fixture_data = json.loads((ROOT / fixture).read_text())
    outputs = list(generate_one.map(fixture_data["cases"], kwargs={"generate_tokens": generate_tokens}))
    result = {"format": "shadow-linux-pirate-golden-v1", "generate_tokens": generate_tokens,
              "model_sha256": fixture_data["model_sha256"], "table_sha256": fixture_data["table_sha256"],
              "outputs": outputs}
    pathlib.Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out} ({len(result['outputs'])} cases)")
