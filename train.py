"""
Example training script with checkpoints and TensorBoard.

Run locally: python train.py
Run on Modal: modal run --detach modal_run.py --exp train
"""

import os
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, save_checkpoint, restore_optimizer
from tensorboard_utils import TBLogger


def train(output_dir=".", epochs=10, batch_size=32):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    print(f"Training config: epochs={epochs}, batch_size={batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Check for existing checkpoint
    checkpoint_prefix = "checkpoint_epoch"
    checkpoint_path, start_epoch = find_latest_checkpoint(output_dir, checkpoint_prefix)
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model)
        restore_optimizer(optimizer, checkpoint_data, device)
        start_epoch = checkpoint_data['step'] + 1
    else:
        start_epoch = 0

    model.to(device)

    # TensorBoard logger
    logger = TBLogger(output_dir, "train")

    # Synthetic data
    X = torch.randn(1000, 64).to(device)
    y = torch.randint(0, 10, (1000,)).to(device)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        avg_loss = epoch_loss / (len(X) // batch_size)
        accuracy = 100.0 * correct / total

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")
        logger.log(epoch, loss=avg_loss, train_acc=accuracy)

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(output_dir, f"{checkpoint_prefix}{epoch}.pt")
        save_checkpoint(checkpoint_path, epoch, model, optimizer)

    logger.close()

    # Save final model
    final_path = os.path.join(output_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    return {"final_loss": avg_loss, "final_acc": accuracy}


if __name__ == "__main__":
    train()
