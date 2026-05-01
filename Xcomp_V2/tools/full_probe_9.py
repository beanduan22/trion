#!/usr/bin/env python3
"""
Comprehensive 9-backend × 879-pattern compat probe.

Designed to run unattended (via nohup) for several hours.

Pipeline per pattern × backend:
  1. Pick a seed StructuralContext that satisfies pat.is_compatible.
  2. Build a single-pattern ONNX graph.
  3. Run the backend (subprocess for crash-prone ones, in-process otherwise).
  4. Classify: pass / fail:msg / crash:reason / unavailable / incompatible.

Intermediate progress is checkpointed every 50 patterns to disk so the run
can be resumed if interrupted.

Output: campaign_v10_results/pattern_compat_v10_9b.json
        (overwrites incrementally; final write is atomic)
        campaign_v10_results/full_probe_9.progress.json (checkpoint)
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
from onnx import helper, TensorProto

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from xcomp_v2.generation.context import StructuralContext
from xcomp_v2.patterns.library import OTPLibrary

ALL_BACKENDS = [
    "onnxruntime", "torchscript", "torch_compile",
    "xla", "tvm", "tensorflow", "openvino", "tflite", "tensorrt",
]
SUBPROC_BACKENDS = {"torch_compile", "openvino", "tvm", "tensorrt",
                    "tensorflow", "tflite"}
# tf/tflite must run CPU-only because TF 2.18's bundled cuDNN clashes with
# torch+cu124's libs in the same env; running them in a fresh subprocess with
# CUDA_VISIBLE_DEVICES="" keeps the GPU stack clean for the other backends.
CPU_ONLY_BACKENDS = {"tensorflow", "tflite"}
SUBPROC_TIMEOUT_S = 90

_SEEDS = [
    StructuralContext(4, [1, 16, 16, 16], "float32", "NCHW"),
    StructuralContext(4, [1, 32, 16, 16], "float32", "NCHW"),
    StructuralContext(3, [1, 16, 64],     "float32", "NLC"),
    StructuralContext(3, [1, 32, 128],    "float32", "NLC"),
    StructuralContext(2, [1, 64],         "float32", "NC"),
    StructuralContext(2, [1, 256],        "float32", "NC"),
]


def _pick_seed(pat):
    for c in _SEEDS:
        if pat.is_compatible(c):
            return c
    return None


def _build_model(instance, ctx) -> onnx.ModelProto:
    inp = helper.make_tensor_value_info("model_input", TensorProto.FLOAT, list(ctx.shape))
    out = helper.make_tensor_value_info(
        instance.output_name, TensorProto.FLOAT,
        list(instance.output_context.shape),
    )
    g = helper.make_graph(instance.nodes, "g", [inp], [out],
                          instance.initializers)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def _backend_inproc(name):
    if name == "onnxruntime":
        from xcomp_v2.oracle.onnxruntime_backend import ONNXRuntimeBackend
        return ONNXRuntimeBackend()
    if name == "torchscript":
        from xcomp_v2.oracle.torchscript_backend import TorchScriptBackend
        return TorchScriptBackend()
    if name == "torch_compile":
        from xcomp_v2.oracle.torch_compile_backend import TorchCompileBackend
        return TorchCompileBackend()
    if name == "xla":
        from xcomp_v2.oracle.xla_backend import XLABackend
        return XLABackend()
    if name == "tvm":
        from xcomp_v2.oracle.tvm_backend import TVMBackend
        return TVMBackend()
    if name == "tensorflow":
        from xcomp_v2.oracle.tf_backend import TFBackend
        return TFBackend()
    if name == "openvino":
        from xcomp_v2.oracle.openvino_backend import OpenVINOBackend
        return OpenVINOBackend()
    if name == "tflite":
        from xcomp_v2.oracle.tflite_backend import TFLiteBackend
        return TFLiteBackend()
    if name == "tensorrt":
        from xcomp_v2.oracle.tensorrt_backend import TensorRTBackend
        return TensorRTBackend()
    raise ValueError(name)


_SUBPROC_RUNNER = """
import sys, json, os, numpy as np, onnx
sys.path.insert(0, %r)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import warnings; warnings.filterwarnings('ignore')
backend_name = sys.argv[1]
def _be(n):
    from xcomp_v2.oracle.torch_compile_backend import TorchCompileBackend
    from xcomp_v2.oracle.openvino_backend       import OpenVINOBackend
    from xcomp_v2.oracle.tvm_backend            import TVMBackend
    from xcomp_v2.oracle.tensorrt_backend       import TensorRTBackend
    from xcomp_v2.oracle.tf_backend             import TFBackend
    from xcomp_v2.oracle.tflite_backend         import TFLiteBackend
    return {'torch_compile': TorchCompileBackend, 'openvino': OpenVINOBackend,
            'tvm': TVMBackend, 'tensorrt': TensorRTBackend,
            'tensorflow': TFBackend, 'tflite': TFLiteBackend}[n]()
be = _be(backend_name)
if not be.is_available():
    print(json.dumps({'status':'unavailable'})); sys.stdout.flush(); sys.exit(0)
print(json.dumps({'status':'ready'})); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    if line == 'EXIT': break
    try:
        job = json.loads(line)
        m   = onnx.load(job['onnx'])
        inp = m.graph.input[0].name
        x   = np.load(job['npy'])
        r   = be.run(m, {inp: x}, optimized=True)
        out = {'status':'pass' if r.ok else 'fail',
               'detail': (r.error or '')[:120]}
    except Exception as e:
        out = {'status':'crash',
               'detail': type(e).__name__ + ':' + str(e)[:100]}
    print(json.dumps(out)); sys.stdout.flush()
""" % str(_REPO_ROOT)


class _Worker:
    def __init__(self, bname):
        self.bname = bname
        sub_env = {**os.environ}
        if bname in CPU_ONLY_BACKENDS:
            sub_env["CUDA_VISIBLE_DEVICES"] = ""
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _SUBPROC_RUNNER, bname],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=sub_env, text=True,
        )
        # Wait for ready / unavailable banner
        try:
            line = self.proc.stdout.readline().strip()
            d = json.loads(line) if line else {"status": "crash"}
        except Exception:
            d = {"status": "crash"}
        self.available = (d.get("status") == "ready")

    def ask(self, onnx_path, npy_path):
        if self.proc.poll() is not None:
            return ("crash", f"worker_dead rc={self.proc.returncode}")
        try:
            self.proc.stdin.write(
                json.dumps({"onnx": onnx_path, "npy": npy_path}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                return ("crash", "worker_eof")
            d = json.loads(line.strip())
            return (d["status"], d.get("detail", ""))
        except (BrokenPipeError, json.JSONDecodeError) as e:
            return ("crash", f"protocol:{type(e).__name__}")

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("EXIT\n"); self.proc.stdin.flush()
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


def _run_subproc(worker: _Worker, model, feed):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf, \
         tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tn:
        onnx.save(model, tf.name)
        np.save(tn.name, list(feed.values())[0])
        try:
            return worker.ask(tf.name, tn.name)
        finally:
            for p in (tf.name, tn.name):
                try: os.unlink(p)
                except FileNotFoundError: pass


def _run_inproc(be, model, feed):
    try:
        r = be.run(model, feed, optimized=True)
        if r.ok:
            return ("pass", "")
        return ("fail", (r.error or "")[:120])
    except Exception as e:
        return ("crash", f"{type(e).__name__}:{str(e)[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(_REPO_ROOT / "campaign_v10_results" / "pattern_compat_v10_9b.json"),
    )
    ap.add_argument(
        "--progress", default=str(_REPO_ROOT / "campaign_v10_results" / "full_probe_9.progress.json"),
    )
    ap.add_argument(
        "--seed-cache", default=str(_REPO_ROOT / "campaign_v10_results" / "pattern_compat_v10.json"),
        help="If set, reuse 'pass'/'fail' rows from this cache for already-probed backends.",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    progress_path = Path(args.progress)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing cache (resume support) ──
    cache = None
    if progress_path.exists():
        print(f"[resume] loading {progress_path}")
        with open(progress_path) as f:
            cache = json.load(f)
        print(f"[resume] backends so far: {cache['backends']}")
    elif Path(args.seed_cache).exists():
        print(f"[seed]   cloning {args.seed_cache}")
        with open(args.seed_cache) as f:
            cache = json.load(f)
    else:
        cache = {"backends": [], "patterns": {}}

    # ── Probe availability ──
    print("[probe] checking backend availability …")
    avail: Dict[str, bool] = {}
    for b in ALL_BACKENDS:
        try:
            avail[b] = bool(_backend_inproc(b).is_available())
        except Exception as e:
            avail[b] = False
            print(f"   {b:14s} raised {type(e).__name__} → unavailable")
    print(f"[probe] live backends: {[b for b,v in avail.items() if v]}")

    # ── Build live library ──
    print("[load] building OTP library …")
    lib = OTPLibrary()
    name_to_pat = {p.name: p
                   for cat in lib.categories
                   for p in lib.patterns_in(cat)}
    print(f"[load] {len(name_to_pat)} live patterns")

    # Make sure every live pattern has an entry in cache
    for pname in name_to_pat:
        cache["patterns"].setdefault(pname, {})
    if args.limit:
        kept = list(cache["patterns"].keys())[:args.limit]
        cache["patterns"] = {k: cache["patterns"][k] for k in kept}
        print(f"[limit] keeping first {args.limit} patterns")

    # Order to probe: in-process first (fast), subprocess last
    in_proc_backends = [b for b in ALL_BACKENDS if b not in SUBPROC_BACKENDS]
    sub_backends = [b for b in ALL_BACKENDS if b in SUBPROC_BACKENDS]

    rng = np.random.default_rng(0)
    t_start = time.time()
    force_reprobe = os.environ.get("FORCE_REPROBE", "1") == "1"
    for bname in in_proc_backends + sub_backends:
        # Skip only if (a) already probed in *this* run (resume) or
        # (b) the user explicitly asked to reuse via FORCE_REPROBE=0.
        already_probed_here = (
            bname in cache["backends"]
            and any(
                isinstance(per.get(bname), dict) and per[bname].get("__from_run") == "current"
                for per in cache["patterns"].values()
            )
        )
        if already_probed_here:
            print(f"[skip] {bname} probed in this run (resume)")
            continue
        if not force_reprobe and bname in cache["backends"]:
            print(f"[skip] {bname} already in seed cache (FORCE_REPROBE=0)")
            continue

        print()
        print(f"=== probe {bname} ===")
        if not avail[bname]:
            n = 0
            for pname in cache["patterns"]:
                cache["patterns"][pname][bname] = {
                    "status": "unavailable",
                    "detail": "is_available() returned False at probe time",
                }
                n += 1
            print(f"  unavailable → marked {n} patterns")
            if bname not in cache["backends"]:
                cache["backends"].append(bname)
            with open(progress_path, "w") as f:
                json.dump(cache, f, indent=2)
            continue

        be = _backend_inproc(bname) if bname not in SUBPROC_BACKENDS else None
        worker = _Worker(bname) if bname in SUBPROC_BACKENDS else None
        if worker is not None and not worker.available:
            print(f"  [{bname}] worker reported unavailable; marking all patterns unavailable")
            for pname in cache["patterns"]:
                cache["patterns"][pname][bname] = {
                    "status": "unavailable",
                    "detail": "worker is_available() returned False",
                    "__from_run": "current",
                }
            worker.close()
            if bname not in cache["backends"]:
                cache["backends"].append(bname)
            with open(progress_path, "w") as f:
                json.dump(cache, f, indent=2)
            continue
        n_pass = n_fail = n_crash = n_skip = 0
        t0 = time.time()
        for i, pname in enumerate(list(cache["patterns"].keys())):
            pat = name_to_pat.get(pname)
            if pat is None:
                cache["patterns"][pname][bname] = {"status": "unknown",
                                                   "detail": "pattern not in live lib"}
                n_skip += 1; continue
            ctx = _pick_seed(pat)
            if ctx is None:
                cache["patterns"][pname][bname] = {"status": "incompatible",
                                                   "detail": "no seed"}
                n_skip += 1; continue
            try:
                inst = pat.instantiate("model_input", ctx, rng, 0)
                if inst is None: raise RuntimeError("instantiate None")
                model = _build_model(inst, ctx)
                feed = {"model_input": rng.standard_normal(ctx.shape).astype(np.float32)}
            except Exception as e:
                cache["patterns"][pname][bname] = {"status": "crash",
                                                   "detail": f"build:{str(e)[:80]}"}
                n_crash += 1; continue

            if bname in SUBPROC_BACKENDS:
                status, detail = _run_subproc(worker, model, feed)
                if status == "crash" and worker.proc.poll() is not None:
                    # Respawn after a crash so we can keep probing.
                    worker.close()
                    worker = _Worker(bname)
            else:
                status, detail = _run_inproc(be, model, feed)

            cache["patterns"][pname][bname] = {
                "status": status, "detail": detail, "__from_run": "current",
            }
            if   status == "pass":  n_pass += 1
            elif status == "fail":  n_fail += 1
            else:                   n_crash += 1

            if (i + 1) % 50 == 0:
                el = time.time() - t0
                rate = (i + 1) / max(el, 0.1)
                eta = (len(cache["patterns"]) - i - 1) / max(rate, 0.01)
                print(f"  [{bname}]  {i+1:4d}/{len(cache['patterns']):4d}  "
                      f"pass={n_pass} fail={n_fail} crash={n_crash} skip={n_skip}  "
                      f"({el:.0f}s, eta={eta/60:.0f}m)")
                with open(progress_path, "w") as f:
                    json.dump(cache, f, indent=2)

        if worker is not None:
            worker.close()
        if bname not in cache["backends"]:
            cache["backends"].append(bname)
        print(f"  {bname:14s} done  pass={n_pass} fail={n_fail} crash={n_crash} skip={n_skip} "
              f"in {(time.time()-t0)/60:.1f}m")
        with open(progress_path, "w") as f:
            json.dump(cache, f, indent=2)

    # ── Recompute safe_patterns (treat 'unavailable' as skip) ──
    new_safe = []
    for pname, per in cache["patterns"].items():
        ok_so_far = True
        for b in cache["backends"]:
            entry = per.get(b, {})
            st = entry.get("status") if isinstance(entry, dict) else None
            if st in ("pass", "diverge", "unavailable"):
                continue
            ok_so_far = False
            break
        if ok_so_far:
            new_safe.append(pname)
    cache["safe_patterns"] = sorted(new_safe)

    cache["summary"] = cache.get("summary", {})
    cache["summary"]["full_probe_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cache["summary"]["wall_time_s"]  = round(time.time() - t_start, 1)

    # Atomic final write
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, out_path)
    print()
    print(f"[done] safe_patterns: {len(new_safe)} / {len(cache['patterns'])}")
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
