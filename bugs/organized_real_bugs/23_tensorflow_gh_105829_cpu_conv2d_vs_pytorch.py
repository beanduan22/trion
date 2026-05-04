def main():
    import os
    import subprocess
    import textwrap

    pyexe = "/home/binduan/miniconda3/envs/xcomp_gpu/bin/python"

    code = textwrap.dedent(
        """
        import numpy as np
        import torch
        import tensorflow as tf
        torch.manual_seed(7)
        np.random.seed(7)
        tf.random.set_seed(7)
        input_tensor = torch.randn([1, 96, 9, 9], dtype=torch.float32)
        weight = torch.randn([64, 96, 3, 3], dtype=torch.float32)
        bias = torch.randn([64], dtype=torch.float32)
        pt_conv = torch.nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0], kernel_size=weight.shape[2:], stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=True)
        with torch.no_grad():
            pt_conv.weight = torch.nn.Parameter(weight)
            pt_conv.bias = torch.nn.Parameter(bias)
            pt_out = pt_conv(input_tensor).cpu().numpy()
        tf_weight = tf.convert_to_tensor(weight.permute(2, 3, 1, 0).numpy())
        tf_bias = tf.convert_to_tensor(bias.numpy())
        tf_conv = tf.keras.layers.Conv2D(filters=weight.shape[0], kernel_size=weight.shape[2:], strides=(1, 1), dilation_rate=(1, 1), padding='SAME', use_bias=True, kernel_initializer=tf.constant_initializer(tf_weight.numpy()), bias_initializer=tf.constant_initializer(tf_bias.numpy()))
        tf_input = tf.convert_to_tensor(input_tensor.numpy().transpose(0, 2, 3, 1))
        tf_out = tf_conv(tf_input).numpy().transpose(0, 3, 1, 2)
        diff = np.abs(pt_out - tf_out)
        print("pt_shape:", tuple(pt_out.shape))
        print("tf_shape:", tuple(tf_out.shape))
        print("max_abs:", float(diff.max()))
        print("mean_abs:", float(diff.mean()))
        print("mismatch_count:", int(np.sum(diff > 1e-5)))
        print("pt_head:", pt_out.reshape(-1)[:6].tolist())
        print("tf_head:", tf_out.reshape(-1)[:6].tolist())
        """
    )
    env = dict(os.environ)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.run([pyexe, "-c", code], capture_output=True, text=True, env=env)
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    for line in lines:
        print(line)
    if proc.returncode != 0:
        print(proc.stderr.strip() or "subprocess failed")
        return 1
    if any(line.startswith("max_abs:") and float(line.split(":", 1)[1].strip()) > 1e-4 for line in lines):
        print("BUG REPRODUCED: TensorFlow CPU Conv2D diverges from PyTorch baseline")
        return 0
    print("not reproduced")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
