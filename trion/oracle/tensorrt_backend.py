"""
TensorRT Target Backend.
Uses ONNX Runtime's TensorRT execution provider as the most portable path.
Falls back to pure TensorRT Python API if the provider is unavailable.
"""
from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import onnx

from .base import BackendBase, BackendResult

logger = logging.getLogger(__name__)


class TensorRTBackend(BackendBase):
    name = "tensorrt"

    def __init__(self, fp16: bool = False) -> None:
        self.fp16 = fp16

    def is_available(self) -> bool:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                return True
            import tensorrt  # noqa: F401
            return True
        except ImportError:
            return False

    def run(
        self,
        model: onnx.ModelProto,
        inputs: Dict[str, np.ndarray],
        optimized: bool = True,
    ) -> BackendResult:
        # Try ORT + TensorRT EP first (easier setup)
        result = self._run_via_ort(model, inputs, optimized)
        if result.ok:
            return result
        # Fall back to direct TRT Python API
        return self._run_via_trt(model, inputs, optimized)

    def _run_via_ort(self, model, inputs, optimized) -> BackendResult:
        try:
            import onnxruntime as ort
            if "TensorrtExecutionProvider" not in ort.get_available_providers():
                return BackendResult(None, "TRT EP not available", crashed=True)

            trt_opts = {
                "trt_fp16_enable":  self.fp16 and optimized,
                "trt_engine_cache_enable": False,
            }
            if not optimized:
                trt_opts["trt_max_partition_iterations"] = 0   # minimal fusion

            opts = ort.SessionOptions()
            sess  = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=opts,
                providers=[
                    ("TensorrtExecutionProvider", trt_opts),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
            input_names = {i.name for i in sess.get_inputs()}
            feed = {k: v.astype(np.float32)
                    for k, v in inputs.items() if k in input_names}
            output = sess.run(None, feed)[0]
            return BackendResult(output)
        except Exception as exc:
            logger.debug("TRT via ORT failed: %s", exc)
            return BackendResult(None, str(exc), crashed=True)

    def _run_via_trt(self, model, inputs, optimized) -> BackendResult:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser  = trt.OnnxParser(network, TRT_LOGGER)

            serialized = model.SerializeToString()
            if not parser.parse(serialized):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                return BackendResult(None, "; ".join(errs), crashed=True)

            config = builder.create_builder_config()
            if self.fp16 and optimized and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if not optimized:
                config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

            serialized_engine = builder.build_serialized_network(network, config)
            runtime = trt.Runtime(TRT_LOGGER)
            engine  = runtime.deserialize_cuda_engine(serialized_engine)
            context = engine.create_execution_context()

            input_name = model.graph.input[0].name
            x = inputs[input_name].astype(np.float32)
            out_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
            output = np.empty(out_shape, dtype=np.float32)

            d_in  = cuda.mem_alloc(x.nbytes)
            d_out = cuda.mem_alloc(output.nbytes)
            cuda.memcpy_htod(d_in, x)
            context.execute_v2([int(d_in), int(d_out)])
            cuda.memcpy_dtoh(output, d_out)
            return BackendResult(output)
        except Exception as exc:
            logger.debug("Direct TRT failed: %s", exc)
            return BackendResult(None, str(exc), crashed=True)
