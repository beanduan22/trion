#!/usr/bin/env python3
"""TVM worker subprocess.

Runs in a Python 3.10 environment with apache-tvm 0.11.1 + onnx 1.15.0
(see README for setup). Communicates with the main (3.13) oracle over
stdio using length-prefixed pickled messages:

  client → worker:  {"onnx_bytes": <bytes>, "feeds": {name: np.ndarray},
                     "opt_level": 0|3, "input_shapes": {name: tuple}}
  worker → client:  {"output": np.ndarray}   OR  {"error": str}

The worker processes requests in a loop and terminates when it reads EOF
from stdin, so a single Python/TVM load cost is amortised across every
model in a campaign.
"""
from __future__ import annotations
import io
import pickle
import struct
import sys
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.contrib.graph_executor import GraphModule


def _read_message(stream) -> Dict[str, Any] | None:
    """Read one length-prefixed pickled message. Returns None on EOF."""
    header = stream.read(4)
    if not header:
        return None
    if len(header) < 4:
        return None
    (n,) = struct.unpack("!I", header)
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write_message(stream, msg: Dict[str, Any]) -> None:
    data = pickle.dumps(msg)
    stream.write(struct.pack("!I", len(data)))
    stream.write(data)
    stream.flush()


def _run_once(req: Dict[str, Any]) -> Dict[str, Any]:
    onnx_bytes: bytes = req["onnx_bytes"]
    feeds: Dict[str, np.ndarray] = req["feeds"]
    opt_level: int = int(req.get("opt_level", 3))
    # Shape dict required by relay.frontend.from_onnx.
    shape_dict = {k: tuple(v.shape) for k, v in feeds.items()}

    model = onnx.load_from_string(onnx_bytes)
    mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
    target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=opt_level):
        built = relay.build(mod, target=target, params=params)
    dev = tvm.cpu(0)
    gm = GraphModule(built["default"](dev))
    for name, arr in feeds.items():
        gm.set_input(name, arr.astype(np.float32))
    gm.run()
    # Return the first output.
    out = gm.get_output(0).numpy()
    return {"output": np.ascontiguousarray(out)}


def main() -> int:
    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            req = _read_message(stdin)
        except Exception as exc:
            _write_message(stdout, {"error": f"read failed: {exc}"})
            return 1
        if req is None:
            return 0   # EOF: parent closed our stdin, exit cleanly.
        try:
            resp = _run_once(req)
        except Exception:
            resp = {"error": traceback.format_exc()[-4000:]}
        try:
            _write_message(stdout, resp)
        except Exception as exc:
            sys.stderr.write(f"tvm_worker: write failed: {exc}\n")
            return 1


if __name__ == "__main__":
    sys.exit(main())
