def main():
    import json
    import os
    import subprocess
    import textwrap

    pyexe = "/home/binduan/miniconda3/envs/xcomp_gpu/bin/python"

    code = textwrap.dedent(
        """
        import json
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import numpy as np
        import tensorflow as tf
        np.random.seed(321)
        tf.random.set_seed(321)
        def run_conv(use_opt):
            inputs = tf.keras.Input(shape=(64, 64, 16), name='input_0')
            x = tf.keras.layers.Conv2D(filters=24, kernel_size=5, padding='same', kernel_initializer='glorot_uniform', bias_initializer='zeros', activation=None)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            test_input = np.random.randn(1, 64, 64, 16).astype(np.float32)
            tf_output = model.predict(test_input, verbose=0)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if use_opt:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            input_idx = interpreter.get_input_details()[0]['index']
            output_idx = interpreter.get_output_details()[0]['index']
            interpreter.set_tensor(input_idx, test_input)
            interpreter.invoke()
            tfl = interpreter.get_tensor(output_idx)
            mae = float(np.mean(np.abs(tf_output - tfl)))
            max_abs = float(np.max(np.abs(tf_output - tfl)))
            return {'mae': mae, 'max_abs': max_abs, 'tf_head': tf_output.reshape(-1)[:6].tolist(), 'tfl_head': tfl.reshape(-1)[:6].tolist()}
        no_opt = run_conv(False)
        opt = run_conv(True)
        ratio = float('inf') if no_opt['mae'] == 0.0 else opt['mae'] / no_opt['mae']
        print(json.dumps({'no_opt': no_opt, 'opt': opt, 'ratio': ratio}))
        """
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    proc = subprocess.run([pyexe, "-c", code], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stderr.strip() or proc.stdout.strip() or "subprocess failed")
        return 1
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    row = json.loads(lines[-1])
    print("expected_no_opt_mae:", row["no_opt"]["mae"])
    print("optimized_mae:", row["opt"]["mae"])
    print("optimized_max_abs:", row["opt"]["max_abs"])
    print("ratio:", row["ratio"])
    print("tf_head:", row["opt"]["tf_head"])
    print("tflite_head:", row["opt"]["tfl_head"])
    if row["opt"]["mae"] > 1e-3 and row["opt"]["max_abs"] > 1e-2:
        print("BUG REPRODUCED: TFLite DEFAULT optimization causes Conv2D precision loss")
        return 0
    print("not reproduced")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
