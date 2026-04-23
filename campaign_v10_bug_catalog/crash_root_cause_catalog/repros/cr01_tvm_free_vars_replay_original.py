#!/usr/bin/env python3
"""
CR-01 · Original-model replay for the 5 TVM free-var crashes from campaign_v10.

Loads the saved ONNX model for each of the 5 crash artifacts (385, 905, 959,
1225, 1452) and replays the TVM build. All five crash with the identical
"contains free variables" error signature — same root cause as in
``cr01_tvm_free_vars_min.py`` (self-ref binop after freeze_params=True build).

Run:
    python crash_root_cause_catalog/repros/cr01_tvm_free_vars_replay_original.py
"""
import os
import sys

MODELS = [385, 905, 959, 1225, 1452]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    try:
        import onnx
        import tvm
        from tvm import relay
    except ImportError:
        print("[setup] missing tvm/onnx in this interpreter."); return 2

    repro = 0
    for mid in MODELS:
        onnx_path = os.path.join(
            ROOT, "campaign_v10_results", "compiler_crashes",
            f"crash_{mid:06d}.onnx",
        )
        if not os.path.exists(onnx_path):
            print(f"[skip] {onnx_path} not found"); continue
        model = onnx.load(onnx_path)
        inp = model.graph.input[0]
        shape = {inp.name: [d.dim_value for d in inp.type.tensor_type.shape.dim]}
        mod, params = relay.frontend.from_onnx(model, shape, freeze_params=True)
        try:
            with tvm.transform.PassContext(opt_level=0):
                relay.build(mod, target="llvm", params=params)
            print(f"model {mid}: build OK — not reproduced")
        except Exception as e:
            txt = str(e)
            if "free variables" in txt or "fv.size()" in txt:
                print(f"model {mid}: REPRO — {txt.rstrip().splitlines()[-1][:160]}")
                repro += 1
            else:
                print(f"model {mid}: other error — {txt[-160:]}")
    print(f"\nreproduced: {repro} / {len(MODELS)}")
    return 0 if repro == len(MODELS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
