import sys

try:
    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(2)

def run_case(name: str, values: list[float]) -> bool:
    x_vals = np.array(values, dtype=np.float32)
    g = helper.make_graph(
        [
            helper.make_node("Cast", ["X"], ["T1"], to=TensorProto.INT32),
            helper.make_node("Cast", ["T1"], ["Y"], to=TensorProto.BOOL),
        ],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [len(x_vals)])],
        [helper.make_tensor_value_info("Y", TensorProto.BOOL, [len(x_vals)])],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    out = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"X": x_vals})[0]
    exp = (x_vals.astype(np.int32) != 0)
    print(f"{name}:")
    print(f"  x={x_vals.tolist()}")
    print(f"  ort={out.tolist()}")
    print(f"  exp={exp.tolist()}")
    return not np.array_equal(out, exp)

def main() -> int:
    bug = False
    bug |= run_case("fractional_large_mix", [-3.9, -0.4, 0.4, 3.9])
    bug |= run_case("tiny_signed_nonzeros", [-1e-12, -1e-20, 1e-20, 1e-12])
    if bug:
        print("BUG REPRODUCED: float->int32->bool truncation bug")
        return 0
    print("not reproduced")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
