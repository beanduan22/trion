def main():
    import torch
    from torch import Tensor

    @torch.no_grad()
    def eager_step(param: Tensor):
        view_param = param[:]
        view_param.add_(2.0)
        return view_param

    @torch.no_grad()
    @torch.compile(backend="aot_eager")
    def compiled_step(param: Tensor):
        view_param = param[:]
        view_param.add_(2.0)
        return view_param

    a = torch.tensor([3.0, 4.0])
    b = torch.tensor([3.0, 4.0], requires_grad=True)
    exp1 = eager_step(a).clone()
    exp2 = eager_step(a).clone()
    print("input:", [3.0, 4.0])
    print("expected_after_call1:", exp1.tolist())
    print("expected_after_call2:", exp2.tolist())
    try:
        out1 = compiled_step(b).clone()
        out2 = compiled_step(b).clone()
        print("compiled_after_call1:", out1.tolist())
        print("compiled_after_call2:", out2.tolist())
        print("not reproduced")
        return 1
    except Exception as e:
        lines = str(e).splitlines()
        detail = next((line for line in lines if "A view was created in no_grad mode" in line), lines[-1] if lines else str(e))
        print("bug_type:", type(e).__name__)
        print("bug:", detail)
        print("BUG REPRODUCED: torch.compile no_grad view inplace failure")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
