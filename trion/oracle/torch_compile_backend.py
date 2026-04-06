"""
torch.compile (Inductor) Target Backend.

Converts ONNX → PyTorch (via onnx2torch) then:
  optimized=True  → torch.compile(backend="inductor", mode="max-autotune-no-cudagraphs")
  optimized=False → plain PyTorch eager (no compilation)

torch.compile is the primary target for finding PyTorch 2.x compiler bugs.
The Inductor backend performs:
  - Kernel fusion
  - Memory layout optimization
  - Auto-tuned triton kernels (if CUDA available)
  - Constant folding and algebraic simplification
"""
from __future__ import annotations
import logging
import concurrent.futures
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

_COMPILE_TIMEOUT = 60  # seconds before aborting compilation

logger = logging.getLogger(__name__)


def _patch_onnx2torch_for_compile() -> None:
    """
    Monkey-patch all onnx2torch converters that call torch.Size(tensor) to be
    torch.compile-compatible.

    Root cause: torch.Size(tensor) iterates the tensor as Python ints — fails when
    the tensor is a FakeTensor during torch.compile tracing. Four converters affected:
      - OnnxReshape._do_reshape     → torch.Size(shape)
      - OnnxExpand.forward          → torch.ones(torch.Size(shape), ...)
      - OnnxTile.forward            → input.repeat(torch.Size(repeats))
      - OnnxConstantOfShape.forward → torch.full(size=torch.Size(shape), ...)

    Fix strategy: replace torch.Size(tensor) with [tensor[i] for i in range(n)]
    where n = tensor.shape[0] (statically known = number of dims).
    Each tensor[i] is a 0-dim symbolic Tensor that torch.compile can trace.
    """
    try:
        import torch
        import onnx2torch.node_converters.reshape as _r
        import onnx2torch.node_converters.expand as _e
        import onnx2torch.node_converters.tile as _t
        import onnx2torch.node_converters.constant_of_shape as _c

        # ── Reshape ──────────────────────────────────────────────────────────
        @staticmethod  # type: ignore[misc]
        def _safe_reshape(input_tensor: "torch.Tensor",
                          shape: "torch.Tensor") -> "torch.Tensor":
            n = shape.shape[0]
            return input_tensor.reshape([shape[i] for i in range(n)])

        _r.OnnxReshape._do_reshape = _safe_reshape

        # ── Expand ───────────────────────────────────────────────────────────
        _orig_expand_forward = _e.OnnxExpand.forward

        def _safe_expand_forward(self, input_tensor, shape):
            if torch.onnx.is_in_onnx_export():
                return _orig_expand_forward(self, input_tensor, shape)
            n = shape.shape[0]
            sym_shape = [shape[i] for i in range(n)]
            return input_tensor.expand(sym_shape)

        _e.OnnxExpand.forward = _safe_expand_forward

        # ── Tile ─────────────────────────────────────────────────────────────
        _orig_tile_forward = _t.OnnxTile.forward

        def _safe_tile_forward(self, input_tensor, repeats):
            if torch.onnx.is_in_onnx_export():
                return _orig_tile_forward(self, input_tensor, repeats)
            n = repeats.shape[0]
            return input_tensor.repeat([repeats[i] for i in range(n)])

        _t.OnnxTile.forward = _safe_tile_forward

        # ── ConstantOfShape ───────────────────────────────────────────────────
        _orig_cos_forward = _c.OnnxConstantOfShape.forward

        def _safe_cos_forward(self, shape):
            n = shape.shape[0]
            sym_shape = [shape[i] for i in range(n)]
            fill_value = self.value.item()
            return torch.full(
                size=sym_shape,
                fill_value=int(fill_value) if isinstance(fill_value, bool) else fill_value,
                dtype=self.value.dtype,
                device=self.value.device,
            )

        _c.OnnxConstantOfShape.forward = _safe_cos_forward

        logger.debug("onnx2torch Reshape/Expand/Tile/ConstantOfShape patched "
                     "for torch.compile compatibility.")
    except Exception as exc:
        logger.warning("Could not patch onnx2torch converters: %s", exc)


# Apply once at import time
_patch_onnx2torch_for_compile()


class TorchCompileBackend(BackendBase):
    name = "torch_compile"

    def __init__(self, mode: str = "default") -> None:
        # mode: "default", "reduce-overhead", "max-autotune-no-cudagraphs"
        self.mode = mode

    def is_available(self) -> bool:
        try:
            import torch
            import onnx2torch  # noqa: F401
            return hasattr(torch, "compile")
        except ImportError:
            return False

    def run(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        optimized: bool = True,
    ) -> BackendResult:
        try:
            import torch
            import onnx2torch

            torch_model = onnx2torch.convert(model)
            torch_model.eval()

            input_name = model.graph.input[0].name
            x = torch.from_numpy(inputs[input_name].astype(np.float32))

            # Move to CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_model = torch_model.to(device)
            x = x.to(device)

            if optimized:
                compiled = torch.compile(
                    torch_model,
                    backend="inductor",
                    mode=self.mode,
                    fullgraph=False,
                )
                def _run_compiled():
                    with torch.no_grad():
                        return compiled(x)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_run_compiled)
                    try:
                        out = fut.result(timeout=_COMPILE_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        raise RuntimeError(
                            f"torch.compile timed out after {_COMPILE_TIMEOUT}s"
                        )
            else:
                # Baseline eager — same model, no compiler
                with torch.no_grad():
                    out = torch_model(x)

            if isinstance(out, (list, tuple)):
                out = out[0]
            return BackendResult(out.detach().cpu().float().numpy())

        except Exception as exc:
            logger.debug("torch.compile (%s) failed: %s",
                         "opt" if optimized else "no-opt", exc)
            return BackendResult(None, str(exc), crashed=True)
