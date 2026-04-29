import os
import sys
import torch

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.mlp import SimpleMLP


def main():
    model = SimpleMLP()

    weight_path = os.path.join(PROJECT_ROOT, "weights", "simple_mlp.pth")


    torch.save(model.state_dict(), weight_path)

    print(f"model weights saved to: {weight_path}")


if __name__ == "__main__":
    main()
    print(os.path.dirname(os.path.abspath(__file__)))