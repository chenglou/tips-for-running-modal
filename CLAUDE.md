# Modal Training Template

Reusable scaffold for running ML experiments on Modal with checkpoint resumption and TensorBoard logging.

## Quick Start

```bash
# Local
source venv/bin/activate
python train.py

# Modal (detached, survives terminal close)
modal run --detach modal_run.py --exp train
```

## Key Files

- `modal_run.py` - Modal wrapper with volumes for checkpoints/outputs
- `requirements-modal.txt` - Minimal deps for Modal (no local CUDA)
- `checkpoint_utils.py` - Save/resume checkpoints (Modal preemption-safe)
- `tensorboard_utils.py` - Real-time TensorBoard logging utility

## Experiment Contract

Experiments must have a `train(output_dir=".")` function:

```python
def train(output_dir="."):
    # Your training code
    # Save checkpoints to output_dir
    # Return result dict
    return {"accuracy": 0.95}
```

## Modal Commands

```bash
modal run --detach modal_run.py --exp <experiment_name>  # Run
modal app logs <app-id>                                   # Monitor
modal volume ls outputs                                   # List files
modal volume get outputs <file> .                         # Download
```

## TensorBoard

```python
from tensorboard_utils import TBLogger
logger = TBLogger(output_dir, "my_exp")
logger.log(step, loss=0.5, train_acc=0.8, test_acc=0.7)
logger.close()
```

Download runs from Modal and view locally:
```bash
modal volume get outputs runs/ .
tensorboard --logdir runs/
```
