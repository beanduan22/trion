#!/usr/bin/env python3
"""
Bug ID     : bug_000424
Source     : Trion campaign v3 (fuzzing)
Compiler   : onnxruntime, openvino, tensorflow, xla
Patterns   : Flatten + Relu6 + CumSum
Root cause : Flatten(axis=2)+MatMul+BN+Clip(Relu6)+CumSum+Conv+Elu
Tolerance  : 0.1

Exit 0 = BUG REPRODUCED  |  Exit 1 = not reproduced  |  Exit 2 = missing deps
"""
from __future__ import annotations
import base64, io, os, sys
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
except ImportError as e:
    print(f"missing deps: {e}"); sys.exit(2)

TOLERANCE = 0.1
BACKENDS = ['onnxruntime', 'openvino', 'tensorflow', 'xla']
OPSET = 17
IR_VERSION = 8
INPUT_NAME = 'model_input'
OUTPUT_NAME = 'n400_conv_out'
INPUT_SHAPE = [1, 3, 32, 32]
OUTPUT_SHAPE = [1, 128, 32, 32]

# ------------------------------------------------------------------
# Initializers (weights/constants from the fuzzer-discovered model)
# ------------------------------------------------------------------
def _inits() -> list:
    i_0 = numpy_helper.from_array(np.array([1, 3, 32, 32], dtype=np.int64).reshape([4]), 'n0_far_b')
    i_1 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "uEFyvmXtsTzV89o+tC4OPSdpCT+aSdA+JLPTPcAxKL41r9s+Dl7gPTiEqr75KpW8desXPfFR4L0R"
        "wkG+56LTvVHygD2XdHU+CrMWvkaKE7+7pns+W77lPUBFQ74YvWC+bTu3PjgvAr9gLY49lIKFPsdU"
        "1T3l5um8wlUGPr69Ib74vIq+5lmrvnjSEj1kKsi+gcpkPsElr77kFKc9zHWavq+cdLzsHMO9oiSO"
        "PmF39j5/ns28RswjPuK0fr0ecy8+qKDfPQUHt73EujE+4/5GPgSxMT5Y/b89khiIPtKEkT3yxHG+"
        "MZw5vrLzdL2ltds8PxcVv9rNPL7reoo9KtowPUOrUr7oKJa93zlcPtMmFT5TMTw+jxljPjuZpj4c"
        "cle+fPT9vfQrVb6gLyS/c9ToPUO5Hj7DKpm8jfU3vnvB+j3zVcK+AjWUvv8Ghr6tyBq+HD4hPwqw"
        "nT7dqBs+O2Q2vsJGLD46qIq9h2fivcKDkL3Zrl6+3He2Pji7Lb7DmJU+UUHGPvyksD0TgFI+E9Ee"
        "vm0sI75d1ww/9RYhv7TvhT2DTjA+l+mUvcWvlT4CbCY9OAm5PitXT7zpt8c7WQOcvdOpBr6bbcc+"
        "OvcavrX9yD5itzI+gWRxvIRABj5r4IC9/T7dPUcIgb6MXKY+mlq2PncnhDxXZwI9OfYSPGoiqr5i"
        "IKY+ueFqvrefnzw7YYO8vOhIvtSQOz6pyKm+iSeUPcnigD0VtRS+dYY8PrUjfr55+MQ8XyePPply"
        "vL1cAeS9baQavsBV1L2Qs8g+eRaOPjozIj9oWtm+d/QqPfZ0tr0WmGu9jvtQvnLifL2vs2q8Vkkv"
        "Pqz+Iz7XIBI/qOUcPu2ggD7Lowm9XhGQvU0YGj2QGBE9RpyyvgtiID6dI7K9Ou0MPvdhHz6lEnS9"
        "MESCvoK4Y77vZpG94gu7PpW4HT6RYP0+RRlzu9h8Xr0NKAU+CNLRvewYK75gzRI+0v27vcGyXz04"
        "S42+/E5cPVXwAL6WgMM9xNDrPHUJTr8P3369lxCTPXmkiTw1DyE+7BhivRvfWr4J8pY+kAIEOkfH"
        "Br2ZYUE+4idzPTsHHD0ltVQ9ZdSkPKTFR77fFj8+kuQePLW0nb0YM3c+BtzcvX292r4+lSs9Vkvt"
        "voITrD6/JSe+2zQ4PmA2I75GR2Y+NtqjPoMJ/L7L0uI83/lTPhwy+z2+sIG9jfTaPlG3NTxkVJK9"
        "4r+XPZlJ2D73rWa+JSVdPf/IP77LPUg+8udsPtAs7r4fzUo+sS/NvUp4RT5ClLe+gyxsPEaVDz5h"
        "HAK/G4VpvsL7ZT4prjm+9tXmvRU8Hb4BUx6+TTp9PeekrT4xqw2+4AP8PZJdOz5tG/w8qJqKPcOB"
        "/L65vZQ9YTEFP2qVoD7DNCI+QDHJPiOClT2c3ZE9EIQLvzequT7Hwi6/dZLJvfTsIj7l7ic+A7nE"
        "PjxXH739D1u+LSoavRl6rj0GIrS9CmiIPkE3mz6qqtQ+lmx7vq40nr10aCO+3eHTvc3ABD98Eqw9"
        "y5aDPtL/Ez6FdqA+rN2APfA0pb5P6YA+Vq+1vTqCc719rTe+FSKKPsFKID5Qepu+XQw6vbRPnj70"
        "6Ko+0Gq+vnPAlT6xmYa9Y1SCPfyw9b3JoJs94p/dO+Lolz01ORG+bGjJvpBLj75xYzw9OqCZPcx5"
        "xD3TjSO+xLHYvoFJiL4RTbE+jJQwPnfmrr1YmsI9VO9Rug3U+7wD8Mw9WxiJvess7D1DArK+XWOl"
        "PVaqqD6+MaS+loq7vQOZrbx8XZY+kKSUPKbcKb4/5jQ+9Y2SvjfijL74TS88srYzPtQznL7Y30q9"
        "xrJMPu5n1r5UH0W7aX2QvcUx777quKy85MGTPREF3j7iFkc9nLdrPnmM2j7Soac+mVRkvt63or1u"
        "u6G+s7NlvnH4b759bdq+4JeNPnbdN72aKyS+/6wpvch0Dr501lw+HVTuO9PvYz4pNYG+d1rKvLze"
        "Nb4rW8Y+K5+BPpQgrbyfWms93LaLPV/jgr3n6Q8/T76FvJGEDb27k5e9QIqkvd3djbzUjzw8He9b"
        "PkzLqz1lGZy9/40xvl111b52Nhq7zxcKvud1hz52oMG+hQnMPVRCHL9uZ8o+ZxtevnJfir2XyxU+"
        "q90MP7mepb0j1SO/PVOGPi+ndb1K1pq+K+leO/faBr3lADI9R4d8vqQhvr4iAGY+H6VcvpCDdz7+"
        "zH4852+kvSDTBr5kE+K+g34qvGrGrr2xaFK+/6MhPgtDDT4DSYU+k9d3PZXKqL11oS6+eDgSvlgx"
        "qr4JGgI+xyJIvnRnEL7UPqI+CdZ0PoRwPr0MZpc+Klv4PYu55j1kjCE/IxEHP+1hmz2m5ZS+eLfD"
        "vj98gD7ryCm943zYPntHJz5Rirq9IZTQPbjQ6zz0usI+gyqEvmuYbr7Mtxw+T4ervcPniz5iF9u+"
        "4J3rvJddIL54gB6+Mjk0vvoo8z64MC0/FZCgPgCGDT6wXZG+a26ovgFeeT56cXq9hWETvuWPnr7s"
        "w/O+nFeUPRYRcL2y5Z89zkksPiRleL2PPK69ECxtPPu5gr2lx8M9bH/uOyHqHz5TsLM9BKgUvvuO"
        "Oj585VU+svdsvluhzj6rSuw9Vn+FukGTBz+y+CI+ftyWvlJTqj3eWRG+YZe3vhHAdT5BEB4+J7EG"
        "vT/ZHb9P2lG9MMTmvdwgkL6rApe9PJ0NPyrSoz6GOcW+Ruqsvf+0pr02gSe9ljTyvpwvhD6ExJ2+"
        "JyyHPtH9gz2j18W94gpuvW0e7j4jdtw+pOQFPnIpnbyydU4/oIZDvi+11b3IN48+79q8Pu5zOT3H"
        "2oc+pG6uPdPSfb1Yu549uzs9vxiUmj4GnhG97kmcPh+4oD0Nigw+6VEYPkwDkr6Ap6w9ThDaPcGG"
        "zD42r6E9l02ePCy5qD4xfBE89PitPcWxcT7O8ui+wSdLPTqaWzslIAC+3KyyvghAiL296ok9zdTL"
        "vUBh+T1axIy+I2hUPgDSkz2AQ90+JKKbviTz8D7HxUM+eSVHPvEYrz5SvT6+spcRPvONIL65NN09"
        "bRyKvqZ8pb6zcNq+96GDvp6SGb49hHe+bbECPfJ5pr2XOS89EJ9GvCRFhz2g0Q6+AG1BPbJqQj6q"
        "mVM9C6IWPpH7xj0slWI+du2Zvh6VVz6zQSe9r40rvc/quL3+SaS7k0wTPjeRGz1zRX2+HsXgPi4N"
        "1b5GJS++5E4kvcHApz4bFfa8O4AZPgdkhL4wNzS+W6tNvpLYHz6JXXg+baVXPWiwiL5z4Co+d73B"
        "vU6xk76ZRWW+zN6ePuA8Z73hMo++iVK3PiccPrwyJ5O+/OIrP+w7gT5mIzw7bULmvdKIg70SIMQ+"
        "Sh+mPb7+nD5cQJM8H6vxPZ3kp723AF++Wy4BvXGq0D5ty7U9MnEbv5mpRz4Y+p0+QJsUve0kuD5E"
        "xYQ+qN8wvYBPvD1WoC+9hSlovsaUiL7LvZ4+zF0rPtKS/L0Fu3++yUDjPd2w87z2MHo+tiuSPqje"
        "KT6caYM+NWeJPdErlT0KGRI/9f1huwrjXz4egIW9ayi8PZGCHb4Uj7E9/6ZnPvwJmru80Go+K6iP"
        "vjGAZz7GlhI9WdCOPRgYe75ukIg+FLNavI0npD7iOBq+hW+Zvvvqvz4ithS+d7t0PoRfDT4sDPY+"
        "3rmKvu7ADb7G1Ow+Yx9IPj9QrTzM2Xy+MgbkurljA76KUN0+GyGGvk4XG76ANGi+C0QPPuEKtj7p"
        "a3A+ldp0vZLjhTyLiZ4+ZQGePjd8kz4Pp2A+TcOyvYlEoz6Ae7I9185gvTACEr7uuTS+h9qMvj7J"
        "kD574ry8zZwtPLbtqT5QNJW+SaatPUIus739CgQ+mP4Fv6xy6j3fomQ+7c1uPhxZRb5Z7Ba+UOwD"
        "vjhEsr6rga49nYf2vlHsJ74+Rc++KRUxP81Opb7tVq29/mYFPufqlb6Dk0G+gtUDPgZ48D51Bwu/"
        "ZvMdvlgYi75EQcQ9i0skveXSebx9KkI/qY2ePq4rUj0ILE0+FAy5PQ7qdj4dZHg+E5FMvj6RIj5n"
        "W24+ntIqPkiKmz2m6nU+3ofJPszQfj1fn1g+bcKrPozKnz1sJdg8Qw+5PvS0rj2k1Wk+Sr4svehO"
        "9j4FOWO+x1wVv4EPvj4NMhc+xhxTPROl1j0Y8Js+6rGMPLE/lT3W6IC93a8FP3MGaT1lv7A9r6GA"
        "vlBcxT5ntIm+ygGhvQFB/j0Z+ts+/fFmvXMGdD46ICO+MZaVPeFHGr2miti9z+JLPae6Eb0YC0C+"
        "fZeKPSs23D5GPvU9gGwSvjgdkj5Lyba9kVa/Pt5nh76QVYi994G2PRcDEb5VhaS9SpBTvYLmgr4W"
        "YvC9Eo9JPveWADus6dE92PD2vl/jwb1hVvQ9KchOPq9Jer5/0gk+0SoRv1CvA74OV8E9nJCoPQq+"
        "X7xFkAS+J1ZgPnrRVr7dlQ6+nosxPij9B77ztIs+nEtqvkDlDL7ydU073TAQPs6FIj6IZns+QXum"
        "Pd+Rgr4fGWs+FVuEPvSuGz5KPs08REGKPlR+gz5tJ64+KbgZPrczZz7KUvu8SFyaPYJXbD2Hss69"
        "KVA+Pu+gzL7JDMO9basFPox2AD88qL+9tysHvhhDO76ptrC99gCBPgppvr6MQeA83QGWPttqW70F"
        "kLO9mt4Nvp36Sj51n8q9416XvUnC4728BwU+NShpvpm8Sb7ERWi+vEfGPEHqFr64Czk+S6RhPhT/"
        "rL4TL/K+U2fmPmh/sb4CtQq+CpKdPZbP+j5LfJ++iBJWPn+zMzzKmRG+nwk+Pd4oLj5Ks66++m7a"
        "vLK9Bj4rM+W+GKGLPkx5Ir6e0O68FSCfvlK7ej07sk64HVNrvtKeiz5UYr8+NZiRviXfZzoCf4u+"
        "ZXgKPvTtrT6OfLM+sP6ovvpfPD5H9p87a25yPVLyszx9Tmq+P5mkvgkOhL25lqu+q38HPrcNNr/g"
        "Ndi+PZkZvrtUMz7yZ0m9mQ6UPre3T728+JE9SfKTPvWhJb4wkh++mJonvq3LT76RcCu+CQYgPf5C"
        "uT5NIc0+fAkoPRNAc759pgC/77ktPh1Fiz431bO9PIZlPuP9Sb5zYmY+XUhFPqOEuz65C6U9SPMf"
        "Por9Wr6mFSi+db7gPqx+CzzbpZE+3uS2vhyrcb3momG+1B8JPnOVHD78IHo+3y81PfZqGz7BoGi+"
        "SO/QPZJ47L0tjLu+VXKGPjxDGT3PIAg+3KOMveWIMLwHWmG+0s8hPiY6hD5KhDA+GWVgO9Oorj4R"
        "5Y8+dDzePdjy1L7LkB0+asd1vjK6JT4aP5q+6YGLvXVKvj6WjGo+1xu5PixwubtvQO4+1UYwPocS"
        "mb5Tchi/gRgeOx+yhr5NSDq/44loPiF5pr5Hmpi+HInFPn4lwT58Hp6+SUXdPIN5xrs8hmY+B5lA"
        "vvIvnL3b5OI96boqvnYPmz6fa/w+Dobiuy+x4D4/sOE+X6abvN1ivT7ikH0+jGbAPg=="
    ), dtype=np.float32).copy().reshape([1, 1, 32, 32]), 'n100_mm4d_w')
    i_2 = numpy_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape([3]), 'n200_bnr6_sc')
    i_3 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape([3]), 'n200_bnr6_b')
    i_4 = numpy_helper.from_array(np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape([3]), 'n200_bnr6_m')
    i_5 = numpy_helper.from_array(np.array([1.0, 1.0, 1.0], dtype=np.float32).reshape([3]), 'n200_bnr6_v')
    i_6 = numpy_helper.from_array(np.array([0.0], dtype=np.float32).reshape([1]), 'n200_bnr6_zero')
    i_7 = numpy_helper.from_array(np.array([6.0], dtype=np.float32).reshape([1]), 'n200_bnr6_six')
    i_8 = numpy_helper.from_array(np.array([3], dtype=np.int64).reshape([]), 'n300_cs_ax')
    i_9 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([128, 3, 3, 3]), 'n400_conv_w')
    i_10 = numpy_helper.from_array(np.frombuffer(base64.b64decode(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    ), dtype=np.float32).copy().reshape([128]), 'n400_conv_b')
    return [i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10]

# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------
def _nodes() -> list:
    return [
        helper.make_node('Flatten', inputs=['model_input'], outputs=['n0_far_fl'], axis=2),
        helper.make_node('Reshape', inputs=['n0_far_fl', 'n0_far_b'], outputs=['n0_far_out']),
        helper.make_node('MatMul', inputs=['n0_far_out', 'n100_mm4d_w'], outputs=['n100_mm4d_out']),
        helper.make_node('BatchNormalization', inputs=['n100_mm4d_out', 'n200_bnr6_sc', 'n200_bnr6_b', 'n200_bnr6_m', 'n200_bnr6_v'], outputs=['n200_bnr6_bn'], epsilon=9.999999747378752e-06),
        helper.make_node('Clip', inputs=['n200_bnr6_bn', 'n200_bnr6_zero', 'n200_bnr6_six'], outputs=['n200_bnr6_out']),
        helper.make_node('CumSum', inputs=['n200_bnr6_out', 'n300_cs_ax'], outputs=['n300_cs_out']),
        helper.make_node('Conv', inputs=['n300_cs_out', 'n400_conv_w', 'n400_conv_b'], outputs=['n400_conv_cv'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Elu', inputs=['n400_conv_cv'], outputs=['n400_conv_out'], alpha=1.0),
    ]

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_model() -> bytes:
    inputs_info  = [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, INPUT_SHAPE)]
    outputs_info = [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, OUTPUT_SHAPE)]
    graph = helper.make_graph(_nodes(), "bug_000424", inputs_info, outputs_info, initializer=_inits())
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model.SerializeToString()


# ------------------------------------------------------------------
# Input tensor (from the failing fuzzer sample)
# ------------------------------------------------------------------
def _input() -> np.ndarray:
    return np.frombuffer(base64.b64decode(
        "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"
    ), dtype=np.float32).copy().reshape([1, 3, 32, 32])


# ------------------------------------------------------------------
# Reference: pure PyTorch eager via onnx2torch
# ------------------------------------------------------------------
def _ref_pytorch(model_bytes: bytes, x: np.ndarray) -> np.ndarray:
    import torch, onnx2torch
    m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
    with torch.no_grad():
        out = m(torch.from_numpy(x))
    if isinstance(out, (list, tuple)): out = out[0]
    return out.detach().cpu().float().numpy().ravel()


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))


# ------------------------------------------------------------------
# Backend drivers
# ------------------------------------------------------------------
def _run_onnxruntime(model_bytes, x):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
        return np.asarray(sess.run(None, {INPUT_NAME: x})[0]).ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_openvino(model_bytes, x):
    try:
        import openvino as ov
        core = ov.Core()
        compiled = core.compile_model(core.read_model(io.BytesIO(model_bytes)), "CPU")
        out = compiled([x])
        return np.asarray(list(out.values())[0]).ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_tensorflow(model_bytes, x, *, jit=False):
    try:
        sys.path.insert(0, "/home/binduan/myspace/trion")
        from trion.oracle.tf_backend import TFBackend
        r = TFBackend().run(onnx.load_from_string(model_bytes), {INPUT_NAME: x}, optimized=jit)
        if r.output is None: return None, (r.error or "run returned None")[:120]
        return r.output.ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_xla(model_bytes, x):
    return _run_tensorflow(model_bytes, x, jit=True)


def _run_torchscript(model_bytes, x):
    try:
        import torch, onnx2torch
        m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
        t = torch.from_numpy(x)
        scripted = torch.jit.trace(m, (t,))
        frozen = torch.jit.optimize_for_inference(torch.jit.freeze(scripted))
        with torch.no_grad():
            out = frozen(t)
        if isinstance(out, (list, tuple)): out = out[0]
        return out.detach().cpu().float().numpy().ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _run_torch_compile(model_bytes, x):
    try:
        import torch, onnx2torch
        m = onnx2torch.convert(onnx.load_from_string(model_bytes)).eval()
        compiled = torch.compile(m, mode="reduce-overhead", fullgraph=False)
        with torch.no_grad():
            out = compiled(torch.from_numpy(x))
        if isinstance(out, (list, tuple)): out = out[0]
        return out.detach().cpu().float().numpy().ravel(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


RUNNERS = {
    "onnxruntime":   _run_onnxruntime,
    "openvino":      _run_openvino,
    "tensorflow":    _run_tensorflow,
    "xla":           _run_xla,
    "torchscript":   _run_torchscript,
    "torch_compile": _run_torch_compile,
}


def main() -> int:
    try:
        model_bytes = build_model()
        x = _input()
        ref = _ref_pytorch(model_bytes, x)
    except Exception as e:
        print(f"setup failed: {type(e).__name__}: {e}"); return 2

    print(f"Bug ID:    {__doc__.splitlines()[2].split(':',1)[1].strip()}")
    print(f"Backends:  {BACKENDS}")
    print(f"Tolerance: {TOLERANCE}")

    any_bug = False
    for backend in BACKENDS:
        run = RUNNERS.get(backend)
        if run is None:
            print(f"  [{backend}] no driver"); continue
        out, err = run(model_bytes, x)
        if err is not None:
            print(f"  [{backend}] CRASH: {err}   →   BUG REPRODUCED")
            any_bug = True
            continue
        diff = _rel_l2(out, ref)
        verdict = "REPRODUCED" if diff > TOLERANCE else "ok"
        print(f"  [{backend}] rel_L2 vs pytorch_ref = {diff:.4e}   →   {verdict}")
        if diff > TOLERANCE:
            any_bug = True

    PASS = not any_bug
    print(f"PASS={PASS}")
    if not PASS:
        print("BUG REPRODUCED"); return 0
    print("not reproduced"); return 1


if __name__ == "__main__":
    sys.exit(main())
