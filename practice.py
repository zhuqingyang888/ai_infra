import time
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def benchmark_forward(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    in_dim: int = 128,
    warmup: int = 10,
    runs: int = 100,
) -> None:
    """
    对指定 device 和 batch_size 做前向推理计时。
    """
    model = model.to(device)
    model.eval()

    x = torch.rand(batch_size, in_dim, device=device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # GPU 计时前后要同步，避免异步影响结果
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            y = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / runs

    print("-" * 60)
    print(f"device       : {device}")
    print(f"batch_size   : {batch_size}")
    print(f"input shape  : {tuple(x.shape)}")
    print(f"output shape : {tuple(y.shape)}")
    print(f"avg time     : {avg_ms:.4f} ms")


def main() -> None:
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    # 先定义模型
    model = SimpleMLP(in_dim=128, hidden_dim=256, out_dim=64)

    # 比较不同 batch
    batch_sizes = [1, 8, 16]

    # 一定会测 CPU
    cpu_device = torch.device("cpu")
    print("\n=== CPU Benchmark ===")
    for bs in batch_sizes:
        benchmark_forward(model, cpu_device, bs)

    # 如果 GPU 可用，再测 GPU
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        print("\n=== GPU Benchmark ===")
        for bs in batch_sizes:
            benchmark_forward(model, gpu_device, bs)
    else:
        print("\nGPU 不可用，已跳过 GPU Benchmark。")


if __name__ == "__main__":
    main()