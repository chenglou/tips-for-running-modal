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

## Gotchas

- **Ignore list must be hardcoded** in `add_local_dir()`. Don't try to read from .gitignore dynamically - the image definition runs on Modal's build server before files are uploaded (chicken-and-egg).
- **Load checkpoint BEFORE `torch.compile()`** - compiled models have `_orig_mod.` prefix in state dict keys. Load into uncompiled model first, then compile.
- **Config is required for checkpoints** - `load_checkpoint()` and `save_checkpoint()` require a config dict. This prevents silently resuming with wrong hyperparameters (e.g., changing lr and accidentally loading old checkpoint).
- **Use unique checkpoint prefix per experiment** - e.g., `CHECKPOINT_PREFIX = "exp_name_checkpoint_step"`. Prevents loading wrong experiment's checkpoint from shared volume.
- **Volumes persist indefinitely** - Old checkpoints from previous runs stick around. Check with `modal volume ls outputs`. Delete stale ones with `modal volume rm outputs <file>` if you change config.

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

**Live monitoring**: Re-run `modal volume get outputs runs/ .` periodically while training. TensorBoard auto-detects new data.
