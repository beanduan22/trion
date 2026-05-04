def main():
    import os
    import subprocess
    import textwrap

    pyexe = "/home/binduan/miniconda3/envs/xcomp_gpu/bin/python"

    code = textwrap.dedent(
        """
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        class MemoryNetwork(tf.keras.layers.Layer):
            def __init__(self, memory_size=16, memory_dim=8, output_dim=4):
                super().__init__()
                self.memory_size = memory_size
                self.memory_dim = memory_dim
                self.output_dim = output_dim
                self.input_projection = tf.keras.layers.Dense(memory_dim)
                self.output_layer = tf.keras.layers.Dense(output_dim)
            def build(self, input_shape):
                self.memory = self.add_weight(shape=(self.memory_size, self.memory_dim), initializer='random_normal', trainable=True, name='memory')
            @tf.function
            def attention(self, query, memory):
                query_expanded = tf.expand_dims(query, axis=1)
                memory_expanded = tf.expand_dims(memory, axis=0)
                scores = tf.reduce_sum(query_expanded * memory_expanded, axis=-1)
                attention_weights = tf.nn.softmax(scores, axis=-1)
                attention_output = tf.matmul(attention_weights, memory)
                return (attention_output, attention_weights)
            def call(self, x, training=False):
                query = self.input_projection(x)
                attention_output, attention_weights = self.attention(query, self.memory)
                combined = query + attention_output
                return self.output_layer(combined)
        class TestModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.memory_network = MemoryNetwork()
            def call(self, x, training=False):
                return self.memory_network(x, training=training)
        x = tf.random.normal([6, 12])
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
    bug_type = next((line.split("child_bug_type:", 1)[1].strip() for line in lines if line.startswith("child_bug_type:")), "NotImplementedError")
    print(f"bug_type: {bug_type}")
    print(f"bug: {detail}")
    print("BUG REPRODUCED: TensorFlow XLA nested tf.function inside Layer.call fails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
