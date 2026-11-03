import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def state_key(camera_info, logic_name, analysis_params=None):
    identity = [logic_name, str(camera_info.deviceId), str(camera_info.presetId), analysis_params or {}]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@contextmanager
def locked_state(camera_info, logic_name, analysis_params=None):
    root = os.getenv("LOGIC_STATE_DIR")
    if not root and os.getenv("KUBERNETES_SERVICE_HOST"):
        raise RuntimeError("K8s requires LOGIC_STATE_DIR on a shared read-write volume")
    key = state_key(camera_info, logic_name, analysis_params)
    directory = Path(root or "./Picture_pkl").resolve() / key
    directory.mkdir(parents=True, exist_ok=True)

    # Keep this inode: deleting the lock would let new callers bypass waiting callers.
    with (directory / ".lock").open("a+b") as lock:
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            metadata_path = directory / "metadata.json"
            if not metadata_path.exists():
                metadata = {
                    "stateKey": key,
                    "label": logic_name,
                    "deviceId": str(camera_info.deviceId),
                    "presetId": str(camera_info.presetId),
                    "analysisParams": analysis_params or {},
                }
                atomic_write(metadata_path, json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))
            yield directory
        finally:
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_write(path, contents):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
