import sys

try:
    import numpy as np
    import onnx
    import onnx2torch
    import onnxruntime as ort
    import torch
    from onnx import TensorProto as TP, helper as oh, numpy_helper as onh
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, shape: tuple[int, ...]) -> bool:
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    s = np.array(shape, dtype=np.int64)
    nodes = [
        oh.make_node("Add", ["X", "X"], ["a1"]),
        oh.make_node("Add", ["X", "X"], ["a2"]),
        oh.make_node("Sub", ["a1", "a2"], ["sub"]),
        oh.make_node("Reshape", ["sub", "s"], ["Y"]),
    ]
    g = oh.make_graph(
        nodes,
        "g",
        [oh.make_tensor_value_info("X", TP.FLOAT, list(shape))],
        [oh.make_tensor_value_info("Y", TP.FLOAT, list(shape))],
        initializer=[onh.from_array(s, "s")],
    )
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 8
    mb = m.SerializeToString()
    ref = ort.InferenceSession(mb, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
    net = onnx2torch.convert(onnx.load_from_string(mb)).eval()
    x_t = torch.from_numpy(x)
    with torch.no_grad():
        eager = net(x_t).numpy()
    assert np.allclose(ref, eager)
    try:
        compiled = torch.compile(net, mode="default")
        with torch.no_grad():
            got = compiled(x_t).numpy()
        print(f"{name}: compiled_ok diff={float(np.abs(got - eager).max())}")
        return False
    except Exception as e:
        print(f"{name}: BUG {type(e).__name__}: {str(e)[:180]}")
        return True

def main() -> int:
    bug = False
    bug |= run_case("shape_8x32", (8, 32))
    bug |= run_case("shape_2x3x16", (2, 3, 16))
    if bug:
        print("BUG REPRODUCED: FakeTensor runtime-reshape crash")
        return 0
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
