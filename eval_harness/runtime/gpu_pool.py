"""Host GPU availability and advisory leases shared by independent queue processes."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def gpu_status() -> dict[str, dict]:
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    active = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    result = {}
    for line in inventory.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(",")]
        result[index] = {"uuid": uuid, "memory_mib": int(memory), "has_compute_process": uuid in active}
    return result


class GpuLease:
    """OS-released file lock, so a killed queue cannot leave a permanent reservation.

    Persistent lock files intentionally remain: unlinking them would let processes
    lock different inodes for the same physical GPU. NVML checks still protect jobs
    that do not participate in this advisory-lock protocol.
    """
    def __init__(self, directory: str | Path, gpu: str, owner: str):
        self.path = Path(directory) / f"gpu-{gpu}.lock"
        self.owner = owner
        self.stream = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if stream.seek(0, 2) == 0:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "owner": self.owner}).encode())
        stream.flush()
        return True

    def release(self) -> None:
        if self.stream is None:
            return
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None
