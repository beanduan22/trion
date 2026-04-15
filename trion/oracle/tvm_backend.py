"""TVM Backend — real Apache TVM via a Python 3.10 subprocess worker.

Apache TVM (0.11.1 currently) does not ship a Python 3.13 wheel, and its
source build needs LLVM + onnx<1.16. To keep the main oracle on Python
3.13 while still running genuine TVM, this backend spawns a long-lived
worker subprocess (tools/tvm_worker.py) in a pre-provisioned 3.10 env
and communicates over stdio with length-prefixed pickled messages.

  optimized=True  → relay.build with PassContext(opt_level=3)
  optimized=False → relay.build with PassContext(opt_level=0)

Environment override:
  TRION_TVM_WORKER_PYTHON  — full path to the 3.10 interpreter that has
                             apache-tvm + numpy<2 + onnx<1.16 installed.
                             Defaults to the `clawwork` conda env on this
                             machine; users on other boxes should export
                             this.

If the worker python is missing or the worker fails to start, is_available()
returns False — the oracle then automatically drops this backend from the
campaign.
"""
from __future__ import annotations
import logging
import os
import pickle
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)

# Default worker python — overridable via env var. clawwork is a
# pre-existing 3.10 conda env on this machine that now has apache-tvm +
# numpy<2 + onnx<1.16 installed.
_DEFAULT_WORKER_PY = "/home/binduan/miniconda3/envs/clawwork/bin/python"
_WORKER_SCRIPT = str(
    Path(__file__).resolve().parents[2] / "tools" / "tvm_worker.py"
)

# Hard cap on how long a single TVM build+run may take before we treat
# the worker as hung and restart it. relay.build on a large attention
# graph can be slow, so we err on the generous side.
_RUN_TIMEOUT_SEC = 120.0


class TVMBackend(BackendBase):
    name = "tvm"

    def __init__(self, target: str = "llvm", opt_level: int = 3) -> None:
        self.target    = target
        self.opt_level = opt_level
        self._worker: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._worker_py = os.environ.get(
            "TRION_TVM_WORKER_PYTHON", _DEFAULT_WORKER_PY,
        )

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Start a worker lazily. Fails if the configured 3.10 python
        or tvm_worker.py isn't present, or if TVM fails to import in the
        worker."""
        if not os.path.isfile(self._worker_py):
            return False
        if not os.path.isfile(_WORKER_SCRIPT):
            return False
        try:
            self._ensure_worker()
        except Exception as exc:
            logger.info("TVM worker failed to start: %s", exc)
            return False
        return self._worker is not None and self._worker.poll() is None

    # ── Worker lifecycle ──────────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        self._worker = subprocess.Popen(
            [self._worker_py, _WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
        )

    def _kill_worker(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.kill()
            self._worker.wait(timeout=5.0)
        except Exception:
            pass
        self._worker = None

    # ── Stdio protocol (length-prefixed pickle) ──────────────────────────

    def _send(self, msg: dict) -> None:
        data = pickle.dumps(msg)
        assert self._worker and self._worker.stdin
        self._worker.stdin.write(struct.pack("!I", len(data)))
        self._worker.stdin.write(data)
        self._worker.stdin.flush()

    def _recv(self) -> dict:
        assert self._worker and self._worker.stdout
        header = self._worker.stdout.read(4)
        if len(header) < 4:
            raise EOFError("worker closed stdout")
        (n,) = struct.unpack("!I", header)
        buf = b""
        while len(buf) < n:
            chunk = self._worker.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("worker closed mid-message")
            buf += chunk
        return pickle.loads(buf)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, model: onnx.ModelProto, inputs: Dict[str, np.ndarray],
            optimized: bool = True) -> BackendResult:
        try:
            self._ensure_worker()
        except Exception as exc:
            return BackendResult(None, f"worker start failed: {exc}", crashed=True)

        opt_level = self.opt_level if optimized else 0
        req = {
            "onnx_bytes": model.SerializeToString(),
            "feeds":      {k: np.asarray(v) for k, v in inputs.items()},
            "opt_level":  opt_level,
        }

        # Serialise access to the worker: one request in flight at a time.
        with self._lock:
            t0 = time.time()
            try:
                self._send(req)
            except Exception as exc:
                self._kill_worker()
                return BackendResult(None, f"send failed: {exc}", crashed=True)

            # Enforce a soft timeout. If the worker takes too long, assume
            # it's wedged and restart it.
            deadline = t0 + _RUN_TIMEOUT_SEC
            while True:
                if time.time() > deadline:
                    self._kill_worker()
                    return BackendResult(
                        None, f"TVM worker timed out after {_RUN_TIMEOUT_SEC:.0f}s",
                        crashed=True,
                    )
                if self._worker.poll() is not None:
                    # Worker died unexpectedly — capture stderr tail.
                    err = (self._worker.stderr.read() or b"").decode(
                        errors="replace"
                    )[-600:]
                    self._worker = None
                    return BackendResult(
                        None, f"TVM worker died: {err}", crashed=True,
                    )
                # Try to read one message with a short stdout poll.
                try:
                    resp = self._recv()
                    break
                except EOFError as exc:
                    self._kill_worker()
                    return BackendResult(None, f"worker EOF: {exc}", crashed=True)
                except Exception as exc:
                    self._kill_worker()
                    return BackendResult(None, f"recv failed: {exc}", crashed=True)

        if "error" in resp:
            # Don't kill worker: exceptions in a single model must not kill
            # the whole subprocess. The worker catches and returns them.
            return BackendResult(None, resp["error"], crashed=True)
        return BackendResult(np.asarray(resp["output"]))

    def __del__(self):
        try:
            self._kill_worker()
        except Exception:
            pass
