"""
Modal wrapper for training script.
Run with: modal run modal_run.py
"""

import modal

app = modal.App("testing-modal")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__/", "*.pyc", ".git/"])
)


@app.function(image=image, gpu="T4", timeout=300)
def run_training(epochs: int = 5):
    import sys
    sys.path.insert(0, "/root/project")
    from train import train

    result = train(epochs=epochs)
    print(f"\nResult: {result}")
    print("\n>>> Function complete - container will now shut down <<<")
    return result


@app.local_entrypoint()
def main():
    result = run_training.remote(epochs=5)
    print(f"Returned from Modal: {result}")
