import sys

try:
    import numpy as np
    import openvino as ov
    from onnx import TensorProto, helper
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, a_vals: list[int], b_vals: list[int]) -> bool:
    a = np.array(a_vals, dtype=np.uint8)
    b = np.array(b_vals, dtype=np.uint8)
    g = helper.make_graph(
        [helper.make_node("Add", ["A", "B"], ["Y"])],
        "g",
        [
            helper.make_tensor_value_info("A", TensorProto.UINT8, [len(a)]),
            helper.make_tensor_value_info("B", TensorProto.UINT8, [len(b)]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.UINT8, None)],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    core = ov.Core()
    cm = core.compile_model(core.read_model(m.SerializeToString(), b""), "CPU")
    out = cm({"A": a, "B": b})[cm.output(0)]
    wrap = ((a.astype(np.int32) + b.astype(np.int32)) % 256).astype(np.uint8)
    print(f"{name}:")
    print(f"  ov={out.tolist()}")
    print(f"  wrap={wrap.tolist()}")
    return not np.array_equal(out, wrap)

def main() -> int:
    bug = False
    bug |= run_case("sparse_overflow_positions", [250, 10, 20, 30], [20, 20, 30, 40])
    bug |= run_case("alternating_boundary", [255, 0, 255, 0], [1, 255, 2, 254])
    if bug:
        print("BUG REPRODUCED: uint8 Add saturates instead of wrapping")
        return 0
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
