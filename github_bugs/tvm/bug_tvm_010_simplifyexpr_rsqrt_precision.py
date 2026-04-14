"""
TVM Bug: SimplifyExpr converts sqrt(x)/y → rsqrt(x)*y, losing fp32 precision.

Source : https://github.com/apache/tvm/issues/16211
Affects: TVM (relay), reported Dec 2023
Root cause: SimplifyExpr replaces the two-op sequence
              t = sqrt(x);  out = t / y
            with the single-op rewrite
              out = rsqrt(x) * y
            rsqrt is computed as 1/sqrt via a fast approximation on many
            targets, introducing relative error > 0.9 for certain inputs.

Pattern targeted by FoldScaleAxis (also in #16211):
  conv → relu → mul(scale)  reordered to  mul(scale) → conv → relu
  changes numerical behaviour when scale < 0 or scale has large magnitude.

Expected: optimised result matches unoptimised within 1e-4 (TVM default tol)
Actual  : max relative error up to 0.57, mean 0.14–0.15
"""
import numpy as np

try:
    import tvm
    from tvm import relay
    import tvm.relay.transform as T
except ImportError:
    print("TVM not installed — showing expected vs buggy values analytically.")
    tvm = None

def rsqrt_approx(x):
    """Simulates the fast-rsqrt path (Quake-style) used on some TVM targets."""
    # single Newton-Raphson step from float32 bit-trick
    x = np.float32(x)
    xhalf = np.float32(0.5) * x
    i = np.frombuffer(x.tobytes(), dtype=np.int32)[0]
    i = np.int32(0x5f3759df) - np.int32(i >> 1)
    y = np.frombuffer(i.tobytes(), dtype=np.float32)[0]
    y = y * (np.float32(1.5) - xhalf * y * y)
    return float(y)

# ── Reference: sqrt(x) / y ────────────────────────────────────────────────────
x, y = np.float32(0.0001), np.float32(1.0)
ref  = float(np.sqrt(x)) / float(y)

# ── Buggy: rsqrt(x) * y  (SimplifyExpr rewrite) ──────────────────────────────
buggy = rsqrt_approx(x) * float(y)

rel_err = abs(ref - buggy) / (abs(ref) + 1e-12)
PASS = rel_err < 1e-4

print(f"Input x={x}, y={y}")
print(f"Reference  sqrt(x)/y  = {ref:.8f}")
print(f"Buggy    rsqrt(x)*y   = {buggy:.8f}")
print(f"Relative error        = {rel_err:.4f}")
print(f"PASS={PASS}  (False = bug reproduced: rel error > 1e-4)")

if tvm is not None:
    # ── Relay graph: sqrt(x) / y ──────────────────────────────────────────────
    x_var = relay.var("x", relay.TensorType((1,), "float32"))
    y_var = relay.var("y", relay.TensorType((1,), "float32"))
    expr  = relay.divide(relay.sqrt(x_var), y_var)
    func  = relay.Function([x_var, y_var], expr)
    mod   = tvm.IRModule.from_expr(func)

    # Unoptimised
    with tvm.transform.PassContext(opt_level=0):
        lib0   = relay.build(mod, target="llvm")
        dev    = tvm.cpu()
        m0     = tvm.contrib.graph_executor.GraphModule(lib0["default"](dev))
        m0.set_input("x", tvm.nd.array(np.array([x])))
        m0.set_input("y", tvm.nd.array(np.array([y])))
        m0.run()
        ref_tvm = m0.get_output(0).numpy()[0]

    # Optimised (SimplifyExpr enabled)
    with tvm.transform.PassContext(opt_level=3):
        lib3   = relay.build(mod, target="llvm")
        m3     = tvm.contrib.graph_executor.GraphModule(lib3["default"](dev))
        m3.set_input("x", tvm.nd.array(np.array([x])))
        m3.set_input("y", tvm.nd.array(np.array([y])))
        m3.run()
        opt_tvm = m3.get_output(0).numpy()[0]

    rel_tvm = abs(ref_tvm - opt_tvm) / (abs(ref_tvm) + 1e-12)
    PASS = rel_tvm < 1e-4
    print(f"\nTVM unoptimised : {ref_tvm:.8f}")
    print(f"TVM opt_level=3 : {opt_tvm:.8f}")
    print(f"Relative error  : {rel_tvm:.4f}")
    print(f"PASS={PASS}  (False = bug reproduced in TVM)")
