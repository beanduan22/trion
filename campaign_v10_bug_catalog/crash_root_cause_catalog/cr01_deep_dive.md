# CR-01 Deep Dive — Refined Root-Cause Analysis

The original write-up of CR-01 was:

> TVM emits multiple `Var(name, …)` nodes for a single input when
> `freeze_params=True` interacts with self-referential binary ops.

This audit **disproves two parts** of that hypothesis and refines the
statement to one that is considerably tighter.

## Tight mechanistic statement (after audit)

> Inside `tvm.relay.build(...)`, when the module contains
> **`relay.nn.dense(x, W, units=None)`** whose output feeds
> **`relay.add(y, y)`** (self-referential Add), the build pipeline's
> internal `FuseOps → LowerTE` path emits a lowered primitive that
> references the top-level input `x` as a free variable. The final
> free-variable check aborts with `TVMError: … contains free variables:
> [Var(x, …)]`.
>
> Manually pre-applying `FuseOps` before `relay.build` avoids the crash
> (the pre-applied `Primitive=1` annotations steer build down a
> different lowering path that does not duplicate the Var).

The ONNX → Relay frontend normally emits `nn.dense` with `units=None`
from `MatMul` (rank-2 inputs), which is why every one of the 5 original
crashes is an ONNX `MatMul → (…) → Add(t, t)` pipeline.

## Ablation matrix

Every cell below is a run of `relay.build(mod, target="llvm", opt_level=0)`
on a single-Function Relay module built directly from Relay-IR (no ONNX),
except where noted. `np.random.seed(0)` throughout.

### Per-op ablation (dense followed by a binary op)

| preceding op        | binary op      | result |
|---------------------|----------------|--------|
| `nn.dense(x, W, units=None)` | `add(y, y)` | **CRASH(free-var)** |
| `nn.dense(x, W, units=None)` | `mul(y, y)` | OK |
| `nn.dense(x, W, units=None)` | `sub(y, y)` | OK |
| `nn.dense(x, W, units=None)` | `div(y, y)` | OK |
| `nn.dense(x, W, units=None)` | `add(y, c)` (const second operand) | OK |
| `nn.dense(x, W, units=None)` | `add(y, x2)` (separate input) | OK |

→ The binary op must be `add`, and its two operands must be the same SSA value.

### Dense-attribute ablation

| `nn.dense` attr set | + `add(y, y)` |
|---------------------|--------------|
| `units=None, out_dtype="float32"` | CRASH |
| `units=None` (no out_dtype)       | CRASH |
| `units=D, out_dtype="float32"`    | OK |
| `units=D` (no out_dtype)          | OK |

→ The necessary attribute is `units=None`. `out_dtype` is irrelevant.

### Preceding-op ablation (with the same `add(y, y)` sink)

| preceding op produces `y` | + `add(y, y)` |
|---------------------------|--------------|
| `relay.nn.dense(x, W, units=None)` | CRASH |
| `relay.nn.dense(x, W, units=D)` (Gemm-style)    | OK |
| `Conv2d`                       | OK |
| `Relu(x)`                      | OK |
| `Cast(x)`                      | OK |
| `Identity`                     | OK |
| `add(x, x)` (no dense upstream) | OK |

→ Triggering requires `nn.dense(units=None)` *in particular*, not just any
upstream op or any contraction.

### `freeze_params` ablation (only meaningful when going through the ONNX frontend)

| ONNX frontend path | `freeze_params` | result |
|--------------------|-----------------|--------|
| MatMul → Add(m, m) | `True`          | CRASH |
| MatMul → Add(m, m) | `False`         | CRASH |

→ `freeze_params=True` is **not necessary**. The original hypothesis is
wrong on this point. (With `freeze_params=False` the TVM frontend can
instead complain about `"Dynamic Split not yet supported"` if a Split is
downstream — orthogonal to CR-01.)

### Pass-by-pass failing-stage bisection

Each stage applied manually to the crashing module via `tvm.relay.transform`:

| Stage | Free-variable introduced? |
|-------|---------------------------|
| `relay.frontend.from_onnx(...)` | No (`main.params=[x]`, body free_vars=`[x]` — normal) |
| `bind_params_by_name(main, params)` | No |
| `transform.InferType()` | No |
| `transform.SimplifyInference()` | No |
| `transform.FoldConstant()` | No |
| `transform.FuseOps(fuse_opt_level=0)` | No |
| `relay.build(module_that_is_already_FuseOps'd)` | **OK** (no crash) |
| `relay.build(module_NOT_yet_FuseOps'd)` | **CRASH** |

→ The pre-pass IR is always clean (one `%x` Var). The duplicate Var
appears only inside `relay.build`'s own pipeline, in the chain of passes
that runs after its internal `FuseOps` and before codegen — i.e. the
**Relay → TE lowering (`LowerTE`) stage**.

Individually disabling `FuseOps`, `FoldConstant`, `SimplifyInference`,
`CanonicalizeOps`, `AlterOpLayout`, `Legalize`, `CanonicalizeCast`,
`DeadCodeElimination`, `PlanDevices`, `ToANormalForm` does **not** stop
the crash. Whatever introduces the duplicate `Var(x)` is not one of
those optional passes — it is in the always-on lowering stack.

## Failing stage — one-line label

**Relay → TE lowering inside `relay.build`**, specifically the path taken
for a fused primitive of shape `{nn.dense(units=None) ; add(%out, %out)}`.
This corresponds to `LowerTE` invoking a compute function that de-aliases
the two `add` inputs and inadvertently references the top-level input
`Var(x)` from both sides.

## Best mechanistic description

- **Not**: "duplicated Relay Vars at the source level."  → Refuted — the
  Relay IR after any named pass has exactly one `Var(x)`.
- **Not**: "broken parameter binding."  → Refuted — `freeze_params=False`
  still crashes.
- **Closest**: **alias-handling failure during TE lowering of a
  self-referential `add` whose input is produced by `nn.dense(units=None)`**.
  The fused-primitive body `add(%p0, %p0)` is lowered to a TE function
  that CSE-expands `%p0` twice; under the specific
  `nn.dense(units=None)` compute path, the expansion re-constructs the
  chain from `Var(x)` up, causing `Var(x)` to appear twice in the free
  set of the lowered PrimFunc.

The `units=D` path and non-`add` binary ops hit a different `LowerTE`
code path and do not show this behaviour.

## Updated CR-01 scope (set of matching bugs)

All 5 original matches still hold; the mechanistic refinement narrows
**why**, not **which**.

| model | ONNX chain that contains the trigger |
|-------|--------------------------------------|
| 385 | `MatMul (rank-2) → Add → Relu → … → Add(base, base)` (self-add on the residual base) |
| 905 | `Softmax → MatMul → MatMul → … → Add(ln_out, ln_out)` |
| 959 | `MatMul → MatMul → Softmax → … → Add(gelu_out, gelu_out)` |
| 1225 | `MatMul → Add → Div (GELU) → … → Add(glu_out, glu_out)` |
| 1452 | `MatMul → MatMul → MatMul → HardSwish → Add(hs_out, hs_out)` |

Every model has at least one `Add(t, t)` whose `t` traces back to a
`MatMul` (rank-2) with no `units=` attribute — exactly the
`nn.dense(units=None)` trigger.

## Minimal repros available

| file | scope | LOC |
|------|-------|----:|
| `repros/cr01_tvm_free_vars_min.py` | 2-node ONNX graph → fails in `relay.build` | ~45 |
| `repros/cr01_tvm_relay_min.py` | Relay-IR only, no ONNX — **stricter** | ~25 |
| `repros/cr01_tvm_free_vars_replay_original.py` | replays the 5 original ONNX models | ~30 |

## Suggested upstream report title

> `TVM 0.11.1: relay.build crashes with "contains free variables" when
> `nn.dense(units=None)` output feeds a self-referential `add(y, y)`;
> pre-applying `FuseOps` works around it`
