def main():
    import torch
    torch._dynamo.config.capture_scalar_outputs = True

    def eager():
        max_size = torch.tensor([640, 960])
        batch_shape = list(max_size)
        return torch.ones(batch_shape)

    @torch.compile(backend="inductor", fullgraph=True)
    def compiled():
        max_size = torch.tensor([640, 960])
        batch_shape = list(max_size)
        return torch.ones(batch_shape)

    exp = eager()
    print("expected_shape:", list(exp.shape))
    print("expected_sum:", float(exp.sum()))
    try:
        with torch._subclasses.fake_tensor.FakeTensorMode():
            got = compiled()
            print("compiled_type:", type(got).__name__)
            print("not reproduced")
            return 1
    except Exception as e:
        print("bug_type:", type(e).__name__)
        print("bug:", str(e).splitlines()[0])
        print("BUG REPRODUCED: torch.compile FakeTensor intlist failure")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
