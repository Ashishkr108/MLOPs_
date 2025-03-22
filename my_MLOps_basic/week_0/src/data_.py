import torch

def create_dataset():
    # Generate sample data
    X = torch.rand((100, 2))  # 100 samples, 2 features each
    y = torch.randint(0, 2, (100,))  # Binary labels (0 or 1)
    return X, y
