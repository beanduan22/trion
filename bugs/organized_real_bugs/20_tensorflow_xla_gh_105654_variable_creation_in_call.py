def main():
    import os
    import subprocess
    import textwrap

    pyexe = "/home/binduan/miniconda3/envs/xcomp_gpu/bin/python"

    code = textwrap.dedent(
        """
        import tensorflow as tf
        class TestModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(16, activation='relu')
                self.output_layer = tf.keras.layers.Dense(1)
            def call(self, inputs):
                embedded_0 = tf.nn.embedding_lookup(tf.Variable(tf.random.normal([3, 8])), tf.cast(inputs[:, 0:1], tf.int32))
                embedded_1 = tf.nn.embedding_lookup(tf.Variable(tf.random.normal([2, 5])), tf.cast(inputs[:, 1:2], tf.int32))
                embedded_2 = tf.nn.embedding_lookup(tf.Variable(tf.random.normal([4, 10])), tf.cast(inputs[:, 2:3], tf.int32))
                combined = tf.concat([embedded_0, embedded_1, embedded_2], axis=-1)
                x = self.dense(combined)
                return self.output_layer(x)
        x = tf.constant([[0, 1, 2], [2, 0, 1], [1, 1, 3], [0, 0, 0]], dtype=tf.int32)
        model = TestModel()
        eager = model(x)
        print("eager_shape:", tuple(eager.shape))
        @tf.function(jit_compile=True)
        def compiled_forward(arg):
            return model(arg)
        out = compiled_forward(x)
        print("compiled_shape:", tuple(out.shape))
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
    stderr = (proc.stderr or "").splitlines()
    tail = next((line.strip() for line in stderr if "tf.function only supports singleton tf.Variables created on the first call" in line), next((line.strip() for line in reversed(stderr) if line.strip()), "subprocess failed"))
    print("bug_type: ValueError")
    print(f"bug: {tail}")
    print("BUG REPRODUCED: TensorFlow XLA variable creation inside call fails under jit_compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
