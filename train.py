"""
Example training script with checkpoints and TensorBoard.

Run locally: python train.py
Run on Modal: modal run --detach modal_run.py --exp train
"""

import os
from checkpoint_utils import atomic_torch_save, find_latest_checkpoint, load_checkpoint, save_checkpoint, restore_optimizer
from tensorboard_utils import TBLogger


def train(output_dir=".", epochs=10, batch_size=32, lr=0.001):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    os.makedirs(output_dir, exist_ok=True)

    # Config dict for checkpoint validation - prevents resuming with wrong hyperparameters
    CONFIG = {
        'experiment': 'train',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
    }

    print(f"Training config: {CONFIG}")
    print(f"Output directory: {output_dir}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_compile = device.type == "cuda"
    print(f"torch.compile enabled: {use_compile}")

    # Log file on volume - poll with: modal volume get outputs log.txt -
    log_path = os.path.join(output_dir, "log.txt")
    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    # Check for existing checkpoint (prefix includes experiment name to avoid collisions)
    checkpoint_prefix = f"{CONFIG['experiment']}_checkpoint_epoch"
    checkpoint_path, _ = find_latest_checkpoint(output_dir, checkpoint_prefix)
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_epoch = checkpoint_data['step'] + 1
    else:
        checkpoint_data = None
        start_epoch = 0

    final_path = os.path.join(output_dir, "model_final.pt")
    if checkpoint_data and start_epoch >= epochs:
        if not os.path.exists(final_path):
            atomic_torch_save(checkpoint_data['model_state_dict'], final_path)
            log(f"Restored missing final model from completed checkpoint: {final_path}")
        log(f"Training already complete at epoch {start_epoch}/{epochs}; no work to resume.")
        return checkpoint_data.get(
            'metrics',
            {'status': 'already_complete', 'last_epoch': checkpoint_data['step']},
        )

    if use_compile:
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if checkpoint_data:
        restore_optimizer(optimizer, checkpoint_data, device)

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
        num_batches = 0

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
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        accuracy = 100.0 * correct / total
        result = {"final_loss": avg_loss, "final_acc": accuracy}

        log(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")
        logger.log(epoch, loss=avg_loss, train_acc=accuracy)

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(output_dir, f"{checkpoint_prefix}{epoch}.pt")
        save_checkpoint(checkpoint_path, epoch, model, optimizer, CONFIG, metrics=result)

    logger.close()

    # Save final model
    final_state_dict = {
        k.replace('_orig_mod.', ''): v
        for k, v in model.state_dict().items()
    }
    atomic_torch_save(final_state_dict, final_path)
    print(f"\nFinal model saved: {final_path}")

    return result


if __name__ == "__main__":
    train()
