#!/usr/bin/env python3
"""
Bug ID     : cross_tflite_fully_connected_mul_fusion_crash
Source     : Cross-framework testing (2026-04-16)
Compiler   : TensorFlow Lite 2.21 — converter crash.
Patterns   : multiply(x, scalar) -> matmul(result, W) -> add(result, bias)
             where scalar has shape [1] and x has shape [1, N]
Root cause : The TFLite converter's "fuse Mul+MatMul+Add into
             fully_connected" pass incorrectly absorbs the pre-matmul
             scalar multiply into the input tensor rather than folding it
             into the weight. After fusion the input is inferred as
             `tensor<1xf32>` (shape [1]) instead of `tensor<1xNxf32>`
             (shape [1,N]), failing the fully_connected validation:
                'tfl.fully_connected' op expect
                'input' num_elements % N == 0
             The constraint N changes with the number of rows in W:
               w1 shape [2,*] → requires num_elements % 2 == 0
               w1 shape [4,*] → requires num_elements % 4 == 0
             The Keras model runs correctly (expected output is returned);
             the converter cannot produce a deployable TFLite flatbuffer.

Test cases :
    V1  x=[1,2], w1 shape [2,1] — original report  (% 2)
    V2  x=[1,2], w1 shape [2,1] — different weights (% 2)
    V3  x=[1,4], w1 shape [4,1] — wider input       (% 4)
    V4  x=[1,2], w1 shape [2,2] — wider output      (% 2)

Exit 0 = BUG REPRODUCED (converter crashed on at least one case)
Exit 1 = not reproduced
Exit 2 = missing deps
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    sys.exit(2)


def make_model(w1, b1, r, m1, m2):
    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__(name="m")
            self.w1 = tf.Variable(w1, dtype=tf.float32)
            self.b1 = tf.Variable(b1, dtype=tf.float32)
            self.r  = tf.Variable(r,  dtype=tf.float32)
            self.m1 = tf.Variable(m1, dtype=tf.float32)
            self.m2 = tf.Variable(m2, dtype=tf.float32)
        def call(self, x):
            x  = x  + self.m1
            x2 = tf.math.multiply(x,  self.r)
            x3 = tf.linalg.matmul(x2, self.w1)
            x4 = tf.math.add(x3,      self.b1)
            x5 = tf.math.multiply(x4, self.r)
            x6 = tf.math.add(x5,      self.m2)
            return tf.math.multiply(x6, self.r)
    return Model()


CASES = [
    ("V1: x=[1,2] w1=[2,1] original",
     [[0.], [0.5]], [-4.], [-7.], [-4.], [1.], [[2., -3.]]),
    ("V2: x=[1,2] w1=[2,1] diff weights",
     [[1.], [2.]], [0.],  [-3.], [-1.], [2.], [[1., 0.5]]),
    ("V3: x=[1,4] w1=[4,1]",
     [[1.], [2.], [3.], [4.]], [-4.], [-7.], [-4.], [1.], [[2., -3., 1., 0.]]),
    ("V4: x=[1,2] w1=[2,2] wider output",
     [[1., 0.], [0., 1.]], [-4., -4.], [-7.], [-4.], [1.], [[2., -3.]]),
]

any_bug = False
for label, w1, b1, r, m1, m2, xv in CASES:
    x = tf.constant(xv, dtype=tf.float32)
    m = make_model(w1, b1, r, m1, m2)
    keras_out = m(x).numpy()
    try:
        tfl = tf.lite.TFLiteConverter.from_keras_model(m).convert()
        itp = tf.lite.Interpreter(model_content=tfl)
        itp.resize_tensor_input(itp.get_input_details()[0]['index'], list(x.shape))
        itp.allocate_tensors()
        itp.set_tensor(itp.get_input_details()[0]['index'], x.numpy())
        itp.invoke()
        tfl_out = itp.get_tensor(itp.get_output_details()[0]['index'])
        match = np.allclose(keras_out, tfl_out)
        print(f"{label}: Keras={keras_out.flatten().tolist()} "
              f"TFLite={tfl_out.flatten().tolist()} match={match}")
    except Exception as e:
        # extract the key error line
        key = next((l for l in str(e).splitlines() if "fully_connected" in l or "num_elements" in l), str(e).splitlines()[0])
        print(f"{label}:\n  Keras ={keras_out.flatten().tolist()}\n"
              f"  TFLite=CONVERTER CRASH: {key.strip()[:120]}")
        any_bug = True
    print()

sys.exit(0 if any_bug else 1)
