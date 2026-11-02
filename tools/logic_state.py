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


@contextmanager
def locked_state(camera_info, logic_name):
    root = os.getenv("LOGIC_STATE_DIR")
    if not root and os.getenv("KUBERNETES_SERVICE_HOST"):
        raise RuntimeError("K8s requires LOGIC_STATE_DIR on a shared read-write volume")
    identity = [logic_name, str(camera_info.deviceId), str(camera_info.presetId)]
    key = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()
    directory = Path(root or "./Picture_pkl").resolve() / key
    directory.mkdir(parents=True, exist_ok=True)

    # Keep this inode: deleting the lock would let new callers bypass waiting callers.
    with (directory / ".lock").open("a+b") as lock:
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
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
