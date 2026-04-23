#!/usr/bin/env python3
"""
CR-01 · Stricter, ONNX-free Relay-IR repro of the TVM free-variable crash.

Dropping the ONNX frontend and `freeze_params=True` shows the trigger is
purely at the Relay level: `nn.dense` with `units=None` whose output feeds
a self-referential `add`.

Ablation (see cr01_deep_dive.md):
  nn.dense(x, W, units=None) + add(y, y)  -> CRASH (contains free vars)
  nn.dense(x, W, units=D)     + add(y, y) -> OK
  nn.dense(x, W, units=None) + mul(y, y)  -> OK    (only add triggers)
  nn.dense(x, W, units=None) + add(y, c)  -> OK    (self-ref required)
  identity(x) -> add(y, y)                -> OK    (dense required)

Run:
    python crash_root_cause_catalog/repros/cr01_tvm_relay_min.py

Exit 0 = crash reproduced, 1 = bug gone.
"""
import sys
import numpy as np

try:
    import tvm
    from tvm import relay
except ImportError:
    print("[setup] TVM not importable (use e.g. miniconda3/envs/clawwork)")
    sys.exit(2)


def main() -> int:
    np.random.seed(0)
    D = 4
    x = relay.var("x", shape=(1, D), dtype="float32")
    W = relay.const(np.random.randn(D, D).astype(np.float32))
    # The precise trigger: units=None. Flip it to units=D → build succeeds.
    y = relay.nn.dense(x, W, units=None, out_dtype="float32")
    body = relay.add(y, y)         # self-ref add — mul/sub/div do NOT trigger
    mod = tvm.IRModule.from_expr(relay.Function([x], body))

    try:
        with tvm.transform.PassContext(opt_level=0):
            relay.build(mod, target="llvm")
    except Exception as e:
        t = str(e)
        if "free variables" in t or "fv.size()" in t:
            tail = t.rsplit("\n", 2)[-1][:200]
            print(f"[REPRO] CR-01 triggered: {tail}")
            return 0
        print(f"[unexpected] different error: {t[-200:]}")
        return 2

    print("[not-repro] build succeeded — the bug may be fixed in this TVM version.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
