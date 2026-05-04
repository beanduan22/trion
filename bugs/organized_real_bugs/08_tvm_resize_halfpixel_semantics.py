import sys

try:
    import numpy as np
    import onnxruntime as ort
    import tvm
    from onnx import TensorProto, helper
    from tvm import relay
    from tvm.contrib import graph_executor
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, shape: int, scale: float, mode: str, coord: str, nearest_mode: str) -> bool:
    x = np.arange(shape * shape, dtype=np.float32).reshape(1, 1, shape, shape)
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, scale, scale])
    node = helper.make_node(
        "Resize",
        ["X", "", "scales"],
        ["Y"],
        mode=mode,
        coordinate_transformation_mode=coord,
        nearest_mode=nearest_mode,
    )
    g = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, shape, shape])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        initializer=[scales],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    ort_out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
    mod, params = relay.frontend.from_onnx(m, shape={"X": list(x.shape)})
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm", params=params)
    gm = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    gm.set_input("X", x)
    gm.run()
    tvm_out = gm.get_output(0).numpy()
    max_abs = float(np.abs(ort_out.astype(np.float64) - tvm_out.astype(np.float64)).max())
    print(f"{name}: max_abs={max_abs}")
    return max_abs > 1e-4

def main() -> int:
    bug = False
    bug |= run_case("shape5_scale1.5_nearest_halfpixel", 5, 1.5, "nearest", "half_pixel", "round_prefer_floor")
    bug |= run_case("shape5_scale1.25_nearest_halfpixel", 5, 1.25, "nearest", "half_pixel", "round_prefer_floor")
    if bug:
        print("BUG REPRODUCED: Resize half-pixel semantic mismatch")
        return 0
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
