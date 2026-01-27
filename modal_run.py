"""
Modal wrapper for running experiments on GPU.

Usage:
    modal run --detach modal_run.py --exp train
    modal run --detach modal_run.py --exp my_experiment

Outputs (checkpoints, logs, TensorBoard) are saved to Modal volume.
"""

import modal

app = modal.App("ml-training")

# Volume for outputs (checkpoints, logs, TensorBoard runs)
outputs_volume = modal.Volume.from_name("outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "runs/", "*.pt"])
)


@app.function(
    image=image,
    gpu="T4",  # Change to "A100", "H100", "H200" for more power
    timeout=24 * 60 * 60,  # 24 hours max
    volumes={
        "/outputs": outputs_volume,
    },
)
def run_training(exp_name: str):
    import sys
    import importlib

    sys.path.insert(0, "/root/project")

    # Dynamically import the experiment module
    exp_module = importlib.import_module(exp_name)

    result = exp_module.train(output_dir="/outputs")

    # Volumes auto-persist on function exit
    print(f"\nResult: {result}")
    print("\nOutputs saved to 'outputs' volume.")
    print("Run 'modal volume ls outputs' to see files.")
    print("Run 'modal volume get outputs <filename>' to download.")

    return result


@app.local_entrypoint()
def main(exp: str = "train"):
    print(f"Running experiment: {exp}")
    result = run_training.remote(exp_name=exp)
    print(f"\nReturned from Modal: {result}")
