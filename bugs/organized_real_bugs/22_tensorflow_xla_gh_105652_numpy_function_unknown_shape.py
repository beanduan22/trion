def main():
    import os
    import subprocess
    import textwrap

    pyexe = "/home/binduan/miniconda3/envs/xcomp_gpu/bin/python"

    code = textwrap.dedent(
        """
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        class TestModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.d1 = tf.keras.layers.Dense(32, activation='relu')
                self.d2 = tf.keras.layers.Dense(16)
                self.d3 = tf.keras.layers.Dense(8)
            def call(self, x):
                x = self.d1(x)
                def custom_grad_fn(a, b):
                    return a * b ** 2
                x = tf.numpy_function(custom_grad_fn, [x, tf.constant(3.0)], tf.float32)
                x = tf.nn.relu(self.d2(x))
                return self.d3(x)
        x = tf.random.normal([3, 32])
        model = TestModel()
        out = model(x)
        print("eager_shape:", tuple(out.shape))
        @tf.function(jit_compile=True)
        def compiled_forward(arg):
            return model(arg)
        try:
            compiled_out = compiled_forward(x)
            print("compiled_shape:", tuple(compiled_out.shape))
        except Exception as e:
            print("child_bug_type:", type(e).__name__)
            print("child_bug:", str(e).splitlines()[0] if str(e).splitlines() else str(e))
            raise
        """
    )
    env = dict(os.environ)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    proc = subprocess.run([pyexe, "-c", code], capture_output=True, text=True, env=env)
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    for line in lines:
        print(line)
    print("expected_compiled=shape like eager")
    if proc.returncode == 0:
        print("not reproduced")
        return 1
    detail = next((line.split("child_bug:", 1)[1].strip() for line in lines if line.startswith("child_bug:")), "subprocess failed")
    bug_type = next((line.split("child_bug_type:", 1)[1].strip() for line in lines if line.startswith("child_bug_type:")), "ValueError")
    print(f"bug_type: {bug_type}")
    print(f"bug: {detail}")
    print("BUG REPRODUCED: TensorFlow XLA tf.numpy_function leaves unknown TensorShape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
