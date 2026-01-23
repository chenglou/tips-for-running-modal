"""
Modal-agnostic training script.
Run locally: python train.py
Run on Modal: modal run modal_run.py
"""

import time


def train(epochs=5, batch_size=32):
    # Lazy import so non-PyTorch users can still import this module
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print(f"Training config: epochs={epochs}, batch_size={batch_size}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tiny MLP
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data
    X = torch.randn(1000, 64).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(X) // batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.2f}s")
    print("Model would be saved here in a real scenario")

    return {"final_loss": avg_loss, "elapsed": elapsed}


if __name__ == "__main__":
    train()
