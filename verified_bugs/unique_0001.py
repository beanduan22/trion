#!/usr/bin/env python3
"""
Bug #0001: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['normalization', 'crms_norm'], ['constant', 'neg_neg_cancel'], ['branch', 'add_layernorm'], ['constant', 'mul_zero_elim'], ['broadcast', 'hard_clamp_norm'], ['normalization', 'layernorm_residual_add']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0001.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0001.onnx")
INPUT = np.frombuffer(bytes.fromhex("b670a13d43710d40113d823f2de0933d33b7f4bea56aac3f6c7db1bfc0a35f401ee382bf11c113be75f1aabfaa4f69bf81dfd33f086f7abf0a8b07beb74069bf154b593f0dbda7bbded28bbfdd81473f7a2a5cbf57cac9bdf70db0becccdfe3e7752923ea72600c0b0e51bbe8224ba3d5f0af6be4e49a0be09d08dbf044ee63e706801bf0c23f9be202c4abf9e99823fb790883f4b8c243d946932bf53ad6a3fb4198a3e03fdf2bedcfdf63f1fdca4bb02c9f13fb08cd33fb2d0363fc15bfd3f7a7a5fbf42193ebfcdbb11c0a0a297bf819bfebeb4e7373f95a98ebed6860ebf3b1b403f2d7e0abfb74dc73f18c7b3bf958ec53efecacbbfdec488bef9cb7b3fc881eb3f1eb1cf3f2d5f3e3fbad6433ee7fd913fac7805c052161c407c4201bf3be1563ffa4f4f3f710455bd052034bf0bca593fd5defebf7c00e2bb8591e43e65b8b63f7905cb3f45f242bd562ed5bdeced81be4a83d33ef96de63d0222a03b17bbf63fc0ef7fbf2d05ab3e92bcdcbf0280043ffee167bcd75334be047bb2be0739f5bf7cf2ae3f0044e83eec5260be5a9473beac39383f598c2a3da3d3df3fbee15f3d7822db3fe5f3b2be99682cbd98ccfebebd42643fe3ee3ebe63ee70be2832d1bf5221b83fa83eaebf872e663f4d562d3bdceb263f417852bf8c3a1c3f1d35ef3e06167c3f30f4c13f0f677a3e36a61e3fe226aa3fef82d93ef374274060c8943f92708dbf313331bf097862bf72d5ae3e7df206bf0903c43fad3fbd3dc62d033eb9af903fce80273f49c33f3f63d9a4bfb5ad5ebf7003f4be5906033f2c9d203ea704633e4d0511bed14534be8a1e89be9e114bbfd6cfe73f5d25c0bfb9e0f43ef30662bff7fe28c0969555c06165ec3da78051be5d0ea03f82382c3fdeea333f25286fbec519cb3eb782a4bf862690be2c2ab3bf242747be3a292fbfc874b5bfe72bac3fbe60a53f6ae5e9bd2c29723e4eefafbe00082f3f0862513e75b1093ffc29a3bfdf3837bffa98e8bee6b216bea94f1abeb34e0d3fef250b3ff6732ebfb5768fbf91208d3f0b8289bffe0c613faca606bf006f0d3f49ae97bf57a1a4bf401366bee10eddbed71710be745897bf33babe3e0841c23d74a77f3e070a92bfb47033bfaca7313fec74563ee2ba0bbe64ba8cbf0aad283fd196e33ef4c8c6bff8c58e3f823cee3fc8e140be01934d40671d9bbf38dc6dbf8b0f03c09955b03f76bb1840d2e073bf7aa7fcbf0a440fbf236b053f34c6953f5f5280bf3e9a3e407405b1bff96a663e10004abfeee202404eb2cdbff0d434be1e053dc09163913fbb97613ff1ec4d3e13ffd5bf61d9bb3faf1daabd931dcd3ded8085bfd74415bf724384bff8400540f0cd9c3f616dd43fbfe77e3d5a4d04c0faac42bfd1abcd3f4185c23fd6c2fa3e29ac4d3f67390940d7c3e73d19dbe2bf841023be03d6323f8fd582be0424903f9fb85bbf0f81433fe0b71bc05f44843febb9363ffac9d13e57a8f3bde42398bead438cbf10015f3e84488ebe99130b4069b6993f66ebe3bf64a80bbdc5d59e3e8d5ef33e23331dbd9d381c3e1a7430bf7723cc3ec1b38c3fc007c2bfdc07cfbe508e0b3eb177b23f5c81ccbe35166cbf47e5c23ed84d29bf0a9379bfa9f0bdbf0c9e8bbf9f0f014014a0a3bfea40283f3fb884be790b82bf0684943c7cde8ebe269090bd19e795be6d54513f0b167bbf264ea0be5f16b2bfc95f0bc0eb11263f56c7213f4b0fc8bef19eb03f120175bf7ce15a3e660ba8bfed43023e9998133f4319ab3fde17843f031e4a3f4ebd7dbf044b16bf94f025beaf78b1bf356e673f8b3ef2bc04286b3f375d4fbe0698c73fb71d2a3f068980bec5ebc63ff808853e5b857fbfaf4baebc9c429ebed632933f15b29b3e1a8b7bbf9bbbb4bfd6c5bbbd901c6f3eb786a1bf5ad8aebee9ce66be3fdc08c0fea5a3bef7fd303f21fae2be7cedfc3f125847be95a007bf77aa423ec672a83e2ef8cc3b4e179bbf70bf2e403fe09e3f1f549ebff2eafdbf88243c3fbebf183f1f39e43f3d80763f3b3de0bf151ffb3d5b75dc3fdf8572be3f4419405f89f1bd947293bf2efa8abf9788dcbe8ad49ebebd0e37bfc5551c3f1f8abdbf8e8a4dbdff64c4bdacffbb3fda8e45bfb91ee3bfb8a188bfd973ff3f6a2286bf1ffca9bffbc5813e3f6b1c3f9dadb73f942505bfc5e401bf605810be5683073f0140bcbfed87183f2e8cf4bd19f3f03ed583873fcd1354bfb6ae053f7bbc223f9140b7bf0a4562be742a173f138718bf78514fbe60e535be813c533ef07191be9f7d1dbef980a03f7d9aa0bdf161e83eb09b70bf145839bf9b59d33e8ebf64bfc202d23e5792793f2297643f5ddf25bf471542bf78e0cfbf0b8c8c3efc79933fbcdc69be71c2c0bf34e39cbe2331fcbf69da33bf770137bf00ca403fe36887bc2dc6e6bc5427813d3e1a44bf1ffbac3f4ff3e538af5592bee388b4bff45292bf7777aabf1752a33e788cda3f1550d8bf9fb2b5be1dd3c33e062d06bf8c40adbfa0d2f43e4d2b43bfcc14f13efe33e0be5405dd3e62fe103f886a0ec0a49f15c0d93b16bdc04f7ebf1f15353f808eb9bef50abbbfb49400be4dc721bfa2ead6be1ab71c3f4e0dd5bf9aec06bfed7e8ebf5e2f973fce47883d3f32ec3f55bcacbe0ff89fbfd6c2ad3eec51923ff470533ff4943abffe1a33bfc281dcbedc99b83f5398903f0cc22c3f7e2e31bfda090840b833933e5ffd473f9c5aac3ea1720b3f400d003f489e923dcefc743e62972cbec967be3f7ed82b40f79fadbdfaf063bd248d5c3f606d6e3f146e953fbb4a603f644b093e6dcd98be7fb624bfaac894bfa4b04b3f4b7c26bff8b0b83f4b7c4abfeb47ecbe2ec4ca3fb229b4bce8188e3f3ee25f3ddaa05b3ecf128bbf8ad401bffea502bf9515a43d3d2964bcfb70a3bf190518bfbfd2c73ec763ff3db41587beb43ae3bdb22385bf0a47244008d7af3fca97d9be7517bb3e2ee401bf7fec333e5d0212bf36a0c8bf4234393fe4e9ae3cf857e0bedabeeb3f29e5923f2bced43eb59a153fc19c183f41fc723ea89ad63f4289d5bfe281543fbf66b1bec83b3cbf64fab3be7945eb3ef711dabe7dab97bfaf990e3f69e022bdd129cf3f1ce6b93fbed4aebf74c1c9bcb3c1ab3fbc82143f848889be11c71c3f0f9e02c0f9cd2fbf8ecc273f6b05b4bf0f3b73be07f9443faccafcbdafffbb3f9fc2993f1b17213f9f4a1c3e4fcdf6bfcad5c33f96602dbeaa6f073f955a82bf3b8aad3d8eaa5ebda2f31cbe74588a3f41e2acbec1be38bf33dff5be29e6823f260ec63e143d97be460af2bee021c73e19cf233f2846f8bf76201040fc85a83dc94e44bdcda4083ed25eb43ceb1f1d3f657bdd3e33fc5b3ee25888bf770c263fc3eee5bf4930693fc97a503f345dae3e4632024005ca313ffd5577bfac936a3e72268b3e0c36cebf4951133fad03dabde698d63e07f5a53f8f92cb3fa4623f3f263b413ffaf1f6beb3718a3de7c9c73d83ea773f48ca3e3e2643ab3d50567d3fa9900a400d41d63fcc45dabea86566bfecaccc3e759514bdd78429bfe7cb413f92859ebfc89f883ddcb3483ef122eb3f1ac483bf9d5583be13189d3f79c2c4bf85ceeebfe01b2a3fd28da8be4b71f93e489baebfbad98e3eb30f9abfbc30b03d95c2d8bf85c381bfe861563ebdd9f13f725956bfc6aa57bfd31603bf3c8a913fa52ed63e76096abff90b2d405926203f76fb6d3e5df04a3e16544ebff5a58dbf4aff87bf1656883db2ab48bd8748dbbd8a6156beef6fd63d2187dcbea4ba2ebebb74bebf706d1e3d8c6da2be2f4fa23f16b80f3f9abd6fbfd87f9abf39a4fcbdd57f823e91385f3f4b86dc3e5854e5bedb11e4bfedb6e83c32cf02c0b3a543bd65311c3fe8744fbd37ea88be753075bdaddd1a4085ec2b3fb6ac70bf68276fbf2272c8be76df26bfabdd103f5fc1303eb7befebef90e963ebfb594bf0ec321bfcbe7dfbf6b5a223fc93c1bbf44ca313f40c1083facc0d5bdfbad4bbe34db513e16c9ee3fa78abfbc3cbbc33e2fda2d3f3b53d2bd6bedee3ee185b33f74e6eabd419dbcbe230b9bbe119902be301dd93f8eeaca3e798d96bf9a14cb3e939cd53e617d89bf225c9cbf0610a53ea6c0f8bf0b45453f825898bffbe13440d12be83d35a737bf0a3b9d3e78dd67bf79e55b3eb26c93bf3c80c4bf32402abf50743e3f23ee0d3fe38d7abf06d77ebfdad798bf40957ebf7f5a883ed240643fb55f813e7d99b4bffe7451bfdd975ebe1af547bf78f2e73e0994de3e4a4c823e0f6eb33e2547933fe5a5aabea39c49bf261ec2bea43d36bf68dc503f7c4032bffe60db3d8a6a11bf7d67863f2a9b9a3ef7f60e3ff16dd83da631b43eef6df53efa1d2e3f6bffbe3e5608cd3f4e6d703f393b0940ad946dbfea9917bfef1987bf3007233f359cb53f3df4253fd86840bf202208bfde2e483faccee3bc38685e3fc537c13ebad2eb3df38e733fccbdfdbeed609a3ff01fb53eb449af3fd86961bff92c18be674d9e3fc6e3c9bfea0dd33ecc85873ec99139bf3353323edacbba3fc78278bf4f4cf03fc03dba3f74afb7bf9c8310be959dfdbf4412e4be7023ab3f472c9a3ea9a5813ff11cdfbe72bd05c00c95dfbe3093353f56a92c3fe7abd63e50efb5bf1a397b3f5473833e9af8fb3ff60af63e7dbe473e932cca3f4a017bbe989ac2be56a0ff3b5550a8bf5fcfa23eea7b28bfefa9fc3e62b804404a96f83eb5bc9b3d08339cbe1ef0cabe68ab27bee826943e9170df3f38a18d3f46764cbfb586ff3fb1f98e3e9e9924be63350c3f0d698bbf611dee3dcb5d00bfae5989be6c83e23e66133e3fb5505c3e8eba1dc08572a63f1768a8be7328b7bf8d1bdbbec4746c3ff150bfbfd587193f20774fbe5c567cbedc84e03e6c34c53ff3bb073f45e5b2bf77d5833e1095993f1de78d3fe2de053ff1c9ddbe1266bd3f6a97de3e6219e53d7b5c4cbf108b873f8d73aa3fd676b23f96110640e16ac2bd997e8d3fb9371ec0afd9a7bfefa3103fcd0113bff1553f4046d987bfa732363fc7eac0bf29642dc062f34bbf746d2abf459ddb3f1e1deabee819e63f6df25d3fb0aca73e8df599be8e81ec3e321601c0830985be98f1ec3e5b0804c02fc7983ea44fbb3fc012d2becba8e83fc7df6dbd3e1ee0bee1591ebfdcb500408ce78fbef6d8d13ea0494bbf0709cabf65ab9bbf7c903b3e7ccb093f12a24c3dcbaac73e1f389e3fcd1089bedb0311bf18140b3f132b4a3f32a4adbe09de023f4e6d003fdada783f2eeb18bfe2be09bf5d5f3f3d277d3abc37feef3e2182fdbee1b41cbfeb611d3f8157eabeea41e03f66b910c0027ddebfdb3abbbfed69683f963c28be19411cbf6e31c8be79dfec3e57bf813fa6caa6bf9642c13ee14900bf709822beb5c3aebfd4275e3fb679323f7617ca3fc3ff90bfe1fefcbee9e9ab3f810722bf3eb20cc05bff39bfa17f2abf4ea004c01b37e5bd6d5a903f5804ac3ff5f865bdb292223f1c14333fc4836f3ff99ec0bf03208b3ef8f0f7bf86e5133faf6a29bf7be491be1b411c3fe6eff93dd675dcbf2f37cfbe01b29cbffde1883d67db0d3fbb87a33fc0e27b3d2946b73f3ce6f6bf111e6abfed8e35bef966563f5f0b5e3f11ef8a3ea781bcbf75c46d3fab0e9cbe273a5ebf58ec42bf48e34d3eebf6a7bf3a5113bf6a00e4be430b06beb406853f87c2314058d5efbefb4b5c3f30e5b63ffe64e63eddaefebe4c2416c06d580d40e1fdf63d43ed343f318ad23e02c2ed3ef9bd2bbe"), dtype=np.float32).reshape([1, 16, 64])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
