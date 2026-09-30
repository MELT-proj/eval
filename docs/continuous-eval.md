# Evaluating checkpoints as they are saved

`scripts/watch_checkpoints.py` watches a folder of SMURF `.ckpt` files. For each new one it submits
the evaluation, records the result, and lists the checkpoints not worth keeping (it does not delete anything).
It is a thin loop over `scripts/evaluate.py`: it writes one config per checkpoint and calls
`evaluate.py run`; jobs, logs and reports are the ones described in [running-evaluations.md](running-evaluations.md).

Config: [configs/watch/smurf.yaml](../configs/watch/smurf.yaml).

```
new step=N.ckpt ─► written? (untouched for settle_minutes)
                 ─► every_n picks it? ─► configs/step-N.yaml ─► evaluate.py run --site … ─► SLURM
results/summary.csv (one column per checkpoint) ◄─ report ◄─ done ◄─ jobs finished
candidates ─► list what retention does not keep; you delete
```

## Commands

```bash
python scripts/watch_checkpoints.py run     CONFIG           # the loop (tmux on a login node); --once = one pass
python scripts/watch_checkpoints.py status  CONFIG           # ledger: step, status, metric, kept or candidate
python scripts/watch_checkpoints.py enqueue CONFIG STEP      # evaluate this one whatever every_n says
python scripts/watch_checkpoints.py candidates CONFIG        # writes delete-candidates.txt; deletes nothing
```

## Some rules

- **Ledger**: `<output_dir>/watch_state.json`. Stopping and restarting the loop loses nothing and
  never evaluates a checkpoint twice. Delete an entry (or use `enqueue`) to redo one.
- **`every_n`** counts checkpoints in the order they were discovered (the 1st, then every n-th).
  Use `enqueue` for the last checkpoint of a run.
- **Retention** keeps: the newest checkpoint (training resumes from it), the last `keep_last`, the
  `keep_best.k` best by `metric` (`benchmark/slice/metric`, as in `results/results_long.csv`), every multiple of `keep_every`, and anything not finished evaluating or failed. The rest is only listed by`candidates` (with the `rm` commands in `<output_dir>/delete-candidates.txt`); the script never deletes.
- A step that fails shows as `failed` with a note; fix it and run `enqueue CONFIG STEP`.
