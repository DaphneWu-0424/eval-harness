#!/usr/bin/env python3
"""Reuse the colleague's frozen SDK/Java image without starting its training stack."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-image", default="seq-grpo-3b:20261111")
    p.add_argument("--image", default="eval-harness/scienceworld-env:20261116")
    p.add_argument("--output", default="/ssd/eval-results/scienceworld-image-20261116.json")
    args = p.parse_args()
    base = json.loads(subprocess.check_output(["docker", "image", "inspect", args.base_image], text=True))[0]
    probe = """import importlib.metadata as m,subprocess,json,importlib.util
from pathlib import Path
versions={n:m.version(n) for n in ['scienceworld','py4j','httpx','PyYAML','sympy']}
assert versions['scienceworld']=='1.2.3',versions
java=subprocess.check_output(['java','-version'],stderr=subprocess.STDOUT,text=True)
assert 'version "11.' in java,java
location=Path(importlib.util.find_spec('scienceworld').origin).parent
jars=[str(p) for p in location.rglob('*.jar')]
assert jars,'ScienceWorld JAR is missing; no online download is permitted'
print(json.dumps({'versions':versions,'java':java,'jars':jars}))
"""
    result = subprocess.check_output([
        "docker", "run", "--rm", "--name", "eh-scienceworld-image-probe-20261116",
        "--label", "eval-harness.role=image-probe", "--network", "none",
        "-e", "NVIDIA_VISIBLE_DEVICES=void", "-e", "CUDA_VISIBLE_DEVICES=",
        "--entrypoint", "python3", base["Id"], "-c", probe,
    ], text=True)
    existing = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, text=True)
    if existing.returncode == 0 and json.loads(existing.stdout)[0]["Id"] != base["Id"]:
        raise RuntimeError("refusing to replace a different existing ScienceWorld image")
    subprocess.run(["docker", "image", "tag", base["Id"], args.image], check=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"image": args.image, "id": base["Id"], "base_image": args.base_image,
                                  "probe": json.loads(result), "runtime": "CPU SDK only; no Trainer or Ray started"}, indent=2), encoding="utf-8")
    print(f"Prepared {args.image}: {base['Id']}")


if __name__ == "__main__":
    main()
