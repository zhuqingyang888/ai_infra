import os
import sys
import time
import torch

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.mlp import SimpleMLP


def load_model(weight_path, device):
    model = SimpleMLP()

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def run_inference(model, x):
    with torch.no_grad():
        y = model(x)
    return y


def benchmark(model, x, device, warmup=10, num_runs=1000):
    # warmup
    for _ in range(warmup):
        _ = run_inference(model, x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(num_runs):
        y = run_inference(model, x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_latency_ms = (end - start) / num_runs * 1000

    return y, avg_latency_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)

    weight_path = os.path.join(PROJECT_ROOT, "weights", "simple_mlp.pth")

    model = load_model(weight_path, device)

    x = torch.randn(1, 10).to(device)

    y, avg_latency_ms = benchmark(
        model=model,
        x=x,
        device=device,
        warmup=10,
        num_runs=100
    )

    pred_class = torch.argmax(y, dim=1)

    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("output:", y)
    print("prediction class:", pred_class.item())
    print(f"average latency: {avg_latency_ms:.4f} ms")


if __name__ == "__main__":
    main()