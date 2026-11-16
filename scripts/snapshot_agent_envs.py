#!/usr/bin/env python3
"""Freeze the installed environment dependencies; bind-mounted data stays external."""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld-source", default="alfworld-sweep")
    p.add_argument("--webshop-source", default="webshop-eval")
    p.add_argument("--tag", default="20261116")
    p.add_argument("--output", default="/ssd/eval-results/environment-images-20261116.json")
    args = p.parse_args()
    manifest = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "images": {}}
    sources = {}
    # Check both sources before creating either image. Existing containers remain untouched.
    for benchmark in ("alfworld", "webshop"):
        source = getattr(args, f"{benchmark}_source")
        record = json.loads(docker("inspect", source))[0]
        if record["State"]["Running"]:
            processes = docker("top", source, "-eo", "pid,args").splitlines()[1:]
            if any(line.split(None, 1)[1].strip() != "sleep infinity" for line in processes):
                raise RuntimeError(f"{source} is not idle; snapshot it after its task finishes")
        tag = f"eval-harness/{benchmark}-env:{args.tag}"
        existing = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True)
        if existing.returncode == 0:
            previous = json.loads(existing.stdout)[0]
            labels = previous["Config"].get("Labels") or {}
            if labels.get("eval-harness.snapshot.source") != record["Id"]:
                raise RuntimeError(f"image tag already belongs to another source: {tag}")
        sources[benchmark] = (record, source, tag, existing.returncode == 0)
    for benchmark, (record, source, tag, exists) in sources.items():
        if not exists:
            docker("commit", "--pause=false", "--change", "LABEL eval-harness.role=environment",
                   "--change", f"LABEL eval-harness.snapshot.source={record['Id']}",
                   "--change", "ENV NVIDIA_VISIBLE_DEVICES=void", source, tag)
        image_id = docker("image", "inspect", "--format", "{{.Id}}", tag)
        manifest["images"][benchmark] = {
            "tag": tag, "id": image_id, "source_container": source,
            "source_container_id": record["Id"], "source_base_image_id": record["Image"],
            "external_mounts_not_in_image": record.get("Mounts", []),
        }
        print(f"{benchmark}: {tag} ({image_id})", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved image provenance: {output}")


if __name__ == "__main__":
    main()
