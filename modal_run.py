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
    # Patterns match from the project root: "*.pt" would skip only top-level checkpoints, so artifacts use "**/".
    .add_local_dir(".", remote_path="/root/project", ignore=[
        "venv/", ".venv/", ".git/",
        "**/__pycache__/", "**/*.pyc", "**/runs/", "**/*.pt", "**/*.pt.*.tmp", "**/*.log", "**/log.txt",
    ])
)


@app.function(
    image=image,
    gpu="T4",  # Change to "A100", "H100", "H200" for more power
    timeout=24 * 60 * 60,  # 24 hours max
    # Restart on worker loss instead of dying; training resumes from the latest
    # checkpoint in /outputs at function start.
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/outputs": outputs_volume,
    },
)
def run_training(exp_name: str):
    import sys
    import importlib

    sys.path.insert(0, "/root/project")

    # A reused container needs an explicit refresh before inspecting checkpoints.
    outputs_volume.reload()
    try:
        # Dynamically import the experiment module
        exp_module = importlib.import_module(exp_name)

        result = exp_module.train(output_dir="/outputs")

        print(f"\nResult: {result}")
        print("\nOutputs saved to 'outputs' volume.")
        print("Run 'modal volume ls outputs' to see files.")
        print("Run 'modal volume get outputs <filename>' to download.")

        return result
    finally:
        # Persist the latest checkpoint/log writes promptly, including on errors.
        outputs_volume.commit()


@app.local_entrypoint()
def main(exp: str = "train"):
    # spawn (fire-and-forget), not .remote(): a .remote() client holds a live connection
    # for the whole run, and when that connection breaks (laptop sleep, network blip)
    # the exiting client CANCELS the input — even under `modal run --detach`. With
    # spawn, no client needs to stay up. Always launch with `modal run --detach`;
    # without --detach the app stops as soon as this entrypoint returns.
    print(f"Running experiment: {exp}")
    call = run_training.spawn(exp_name=exp)
    print(f"Spawned function call: {call.object_id}")
    print("Training continues server-side; poll the experiment's log on the outputs volume.")
