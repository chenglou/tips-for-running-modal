# Modal Training Template

Reusable scaffold for running ML experiments on Modal with checkpoint resumption and TensorBoard logging. Extracted from a real training project ([sotaku](https://github.com/chenglou/sotaku)); every gotcha below was hit for real there.

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

## Launching Jobs

- **Use `spawn()`, not `.remote()`, for long runs.** A `.remote()` call keeps the client attached. If the client errors during laptop sleep or a network change, it can cancel the input even when the command used `--detach`. This failure looks like worker preemption in the logs.
- **Keep one `spawn()` per detached entrypoint invocation.** Modal only guarantees that the last spawned call survives after the entrypoint exits. Launch independent jobs with separate `modal run --detach` commands.
- **Add retries to the remote training function.** `retries=modal.Retries(...)` lets a preempted worker restart the same input; resumable checkpoints let the retry continue instead of starting over.
- **Set an explicit timeout.** Modal's default is too short for most training jobs.
- **Do not pipe a detached launch through `tail`, `head`, or similar commands.** The stream can remain open, so the downstream command waits indefinitely.

## Monitoring

- **The local entrypoint and remote worker have separate lifetimes.** `modal run --detach` may print `App completed` when the entrypoint exits even though the spawned worker is still running.
- **Use `modal container list` for active-worker status.** A disconnected local stream does not imply that the worker stopped.
- **Read active, uncommitted logs with `modal container exec`.** For example, `modal container exec <container-id> cat /outputs/run/train.log`.
- **Separate remote-command options with `--`.** For example, `modal container exec <container-id> -- tail -n 30 /outputs/run/train.log`. Without the separator, Modal can parse `-n` as its own option.
- **Create the local directory before a recursive Volume download.** For example, run `mkdir -p downloaded` before `modal volume get outputs run/ downloaded/`. In CLI 1.2.6, a nonexistent destination can be treated as one file, causing several remote files to overwrite that same path. Check the downloaded file list and checksums; a successful CLI exit is not enough.
- **Use `modal volume get` for binary files.** CLI 1.2.6 decodes `container exec` output as UTF-8, so raw tar/zip output can fail even with exit code zero. When a file is only available in the container, encode it as base64 inside the container and decode it locally, then verify the archive and checksum.
- **Use Volume files as the durable result.** After the worker exits, verify its result or final checkpoint with `modal volume ls` or `modal volume get`. No active container alone does not distinguish success from failure.
- **Use `modal app logs` only for active debugging.** App logs expire with the App; output files on a Volume persist until removed.

## Images, Volumes, And Checkpoints

- **Hardcode the `add_local_dir()` ignore list.** Image definitions run before project files are uploaded, so they cannot derive the list from a local `.gitignore`.
- **Use recursive globs for nested artifacts.** In Modal 1.2.6, `*.pt` excludes root-level checkpoints only; `**/*.pt` excludes checkpoints at every depth. The same applies to logs and bytecode. Check representative paths with `modal.file_pattern_matcher.FilePatternMatcher` rather than assuming `.gitignore` matching rules.
- **Reload before reading and commit after writing.** Reused containers call `Volume.reload()` before checkpoint discovery, and wrappers call `Volume.commit()` before returning.
- **Save resumable state atomically.** Include model, optimizer, step, and config. Write a temporary sibling file, then rename it into the checkpoint namespace.
- **Use a unique checkpoint prefix per experiment.** For example, `CHECKPOINT_PREFIX = "exp_name_checkpoint_step"`.
- **Save selected weights when the metric improves.** Recording `best_accuracy` without the matching state is insufficient when the best step falls between periodic checkpoints.
- **Load before `torch.compile()`.** Compiled state dicts contain `_orig_mod.` prefixes. Load into the uncompiled model, then compile it.

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
