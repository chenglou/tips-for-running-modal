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
- `requirements-modal.txt` - Pinned minimal deps for Modal
- `checkpoint_utils.py` - Save/resume checkpoints using atomic publication
- `tensorboard_utils.py` - Real-time TensorBoard logging utility

## Gotchas

- **Ignore list must be hardcoded** in `add_local_dir()`. Don't try to read from `.gitignore` dynamically. The image definition runs on Modal's build server before files are uploaded.
- **Load checkpoint BEFORE `torch.compile()`**. Compiled models have `_orig_mod.` prefixes in state dict keys. Load into the uncompiled model first, then compile.
- **Config is required for checkpoints**. `load_checkpoint()` and `save_checkpoint()` require a config dict so hyperparameter drift fails loudly.
- **Publish checkpoints atomically**. `save_checkpoint()` writes to a temporary sibling file and renames only complete snapshots into the checkpoint namespace.
- **Use unique checkpoint prefixes per experiment**. Example: `CHECKPOINT_PREFIX = "exp_name_checkpoint_step"`.
- **Refresh and commit Volumes explicitly**. A reused container calls `reload()` before looking for checkpoints, and the wrapper calls `commit()` before the invocation exits.
- **Volumes persist independently of App log retention**. Old checkpoints and log files stay around until you remove them.
- **Launch with `spawn()`, not `.remote()`, for long training runs**. A `.remote()` client holds a live connection for the whole run, and when that connection breaks (laptop sleep, a network blip), the exiting client cancels the input — even under `--detach`. Spawn from the local entrypoint and exit; the run then needs no client at all. This canceled three multi-hour training runs on 2026-07-02, and the logs ("Received a cancellation signal while processing input") look deceptively like worker preemption.
- **Set `retries=modal.Retries(...)` on training functions**. Without a retry policy, a lost GPU worker kills the input permanently instead of restarting it; with retries plus checkpoint resumption, worker loss self-heals.
- **A disconnected client does not imply a stopped job**. With `--detach`, losing the stream cleanly (terminal close) leaves the job running — but see the `spawn()` bullet above: a client that *errors out* while attached can cancel the run on its way down.
- **Prefer volume log files for monitoring detached or finished jobs**. `modal app logs` is useful while the app is active, but the most reliable status source is a log file written to `output_dir`, then queried with `modal volume get`.
- **Don't pipe `modal run --detach` through `tail`, `head`, or similar commands**. The detached stream stays open, so buffering commands hang waiting for EOF.

## Experiment Contract

Experiments must expose a `train(output_dir=".")` function:

```python
def train(output_dir="."):
    # Save checkpoints and logs under output_dir
    return {"accuracy": 0.95}
```

## Modal Commands

```bash
modal run --detach modal_run.py --exp <experiment_name>  # Run
modal volume ls outputs                                   # List checkpoints/logs
modal volume get outputs log.txt /dev/stdout             # Poll progress from volume logs
modal app logs <app-id>                                   # Active-run debugging only
modal volume get outputs <file> .                         # Download artifacts
```

## TensorBoard

```python
from tensorboard_utils import TBLogger
logger = TBLogger(output_dir, "my_exp")
logger.log(step, loss=0.5, train_acc=0.8, test_acc=0.7)  # accuracy stored as percent
logger.close()
```

Download runs from Modal and view locally:

```bash
modal volume get outputs runs/ .
tensorboard --logdir runs/
```

Live monitoring: re-run `modal volume get outputs runs/ .` periodically while training. TensorBoard will pick up appended event files.
