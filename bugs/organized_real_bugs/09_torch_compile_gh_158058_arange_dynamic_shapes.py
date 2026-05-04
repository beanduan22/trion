def main():
    import torch
    torch._dynamo.config.capture_scalar_outputs = True

    def eager(xs):
        x1, x2 = xs
        return torch.arange(x1, x2)

    @torch.compile(backend="eager", fullgraph=False)
    def compiled(xs):
        x1, x2 = xs
        return torch.arange(x1, x2)

    xs = torch.tensor([3, 7])
    exp = eager(xs)
    print("input:", xs.tolist())
    print("expected:", exp.tolist())
    try:
        got = compiled(xs)
        print("compiled:", got.tolist())
        print("not reproduced")
        return 1
    except Exception as e:
        print("bug_type:", type(e).__name__)
        print("bug:", str(e).splitlines()[0])
        print("BUG REPRODUCED: torch.compile arange dynamic-shape failure")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
