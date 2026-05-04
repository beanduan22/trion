def main():
    import torch
    import torch.nn as nn
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(1, 1)
            self.layer1 = nn.ReLU()
            with torch.no_grad():
                self.layer0.weight.fill_(2.0)
                self.layer0.bias.fill_(-1.0)

        def forward(self, x):
            y = self.layer0(x)
            z = self.layer1(y)
            z.copy_(y)
            return z

    x = torch.tensor([[-1.0], [0.5], [3.0]])
    model = Model().eval()
    exp = model(x).detach().cpu().reshape(-1)
    print("input:", x.reshape(-1).tolist())
    print("expected:", exp.tolist())
    try:
        got = torch.compile(model)(x).detach().cpu().reshape(-1)
        print("compiled:", got.tolist())
        print("not reproduced")
        return 1
    except Exception as e:
        lines = str(e).splitlines()
        detail = next((line for line in lines if "one of the variables needed for gradient computation" in line), lines[-1] if lines else str(e))
        print("bug_type:", type(e).__name__)
        print("bug:", detail)
        print("BUG REPRODUCED: torch.compile copy_ inplace autograd failure")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
