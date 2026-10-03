# Evaluating SMURF checkpoints on JUPITER

The watcher is a thin loop over `scripts/evaluate.py`: for each checkpoint it writes one config and calls `evaluate.py run --site jupiter`, so jobs are submitted with the settings in [infra/sites/jupiter.sh](../infra/sites/jupiter.sh). Jobs, logs and reports are the ones described in [running-evaluations.md](running-evaluations.md).

| Part                                            | What                                              |
| ----------------------------------------------- | ------------------------------------------------- |
| [1. Set up](#1-set-up-once)                      | session, code, environment, downloads, smoke test |
| [2. Run the watcher](#2-run-the-watcher)         | evaluate each checkpoint as it is saved           |
| [3. Results](#3-follow-progress-and-get-results) | status, tables, logs, one-off actions             |

## 1. Set up (once)

### 1.1. Session

Repeat this every time you log in, or save it in `~/.bashrc`.

```bash
export JUPITER_ACCOUNT=<project>
jutil env activate -p $JUPITER_ACCOUNT
echo $PROJECT $SCRATCH            # both must print a path
```

### 1.2. Code

```bash
mkdir -p $PROJECT/$USER && cd $PROJECT/$USER
git clone git@github.com:MELT-proj/eval.git eval
git clone <speechllm-url> speechllm
cd eval
git switch jupiter-continuous-eval    # only until this is merged in main
```

### 1.3. Environment

The wheels must be aarch64 builds, so if something does not install, it fails here.

```bash
uv venv --python 3.12 $PROJECT/$USER/venvs/smurf-eval # no uv: pip install --user uv
VIRTUAL_ENV=$PROJECT/$USER/venvs/smurf-eval uv pip install -e ../speechllm
VIRTUAL_ENV=$PROJECT/$USER/venvs/smurf-eval uv pip install -e ".[metrics,analysis]"
source $PROJECT/$USER/venvs/smurf-eval/bin/activate
python -c "import torch, nemo, fbk_speechllm, inspect_ai, melteval; print('ok')"  # must print ok
```

From here on, every command assumes this venv is active (run the `source .../activate` line again after each login).

### 1.4. Downloads

The running nodes have no internet and jobs run with `HF_HUB_OFFLINE=1`, so anything missing from the cache fails the job. Download it here, into `$HF_HOME` (`$SCRATCH/$USER/hf_cache`).

| # | What                                        | Used for                                       |
| - | ------------------------------------------- | ---------------------------------------------- |
| 1 | `utter-project/EuroLLM-9B-Instruct-2512`  | the SMURF's LLM                                |
| 2 | `nvidia/canary-1b-v2`                     | the SMURF's audio encoder (`pretrained_asr`) |
| 3 | AIR-Bench data (`pbcong/AIR-bench`)       | `air-bench`                                  |
| 4 | MCIF audio and references (`FBK-MT/MCIF`) | `mcif`                                       |
| 5 | LibriSpeech test-clean                      | `librispeech`                                |
| 6 | `Qwen/Qwen2-Audio-7B-Instruct`            | baseline (2.3)                                 |
| 7 | `Qwen/Qwen3-Omni-30B-A3B-Instruct`        | baseline (2.3), about 60 GB                    |

```bash
cd $PROJECT/$USER/eval
export HF_HOME=$SCRATCH/$USER/hf_cache HF_HUB_OFFLINE=0     # allow downloads in this shell

# 1-2. The SMURF's models
huggingface-cli download utter-project/EuroLLM-9B-Instruct-2512
huggingface-cli download nvidia/canary-1b-v2
ls $HF_HOME/hub/models--nvidia--canary-1b-v2/snapshots/*/    # must contain canary-1b-v2.nemo

# 3, 5. AIR-Bench and LibriSpeech
python scripts/evaluate.py prefetch configs/eval/smurf.yaml --benchmarks air-bench librispeech

# 6-7. The baselines' models
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct

# 4. MCIF, audio included
python - <<'EOF'
from melteval.dataset import spec_dataset
from melteval.readers.base import reader_for_locator
from melteval.manifest import AudioLocator
for dataset_id, task in [("mcif-short-fixed-en", "chunked_asr"), ("mcif-short-fixed-de", "chunked_st"),
                         ("mcif-short-fixed-it", "chunked_st"), ("mcif-short-fixed-zh", "chunked_st")]:
    ds = spec_dataset("configs/hf/mcif.yaml", dataset_id=dataset_id, task=task)
    for s in ds.samples:
        loc = AudioLocator.from_dict(s.metadata["audio"]); reader_for_locator(loc).load_audio(loc)
    print(dataset_id, len(ds.samples), "ok")      # one "ok" line per dataset
EOF

export HF_HUB_OFFLINE=1
```

### 1.5. Local datasets

Showcased here for FLEUR, but this should be done for each local dataset.

```bash
export LOCAL_DATASETS_DIR=<indexed path>
melteval freeze configs/fleurs24-asr-test.yaml -o $PROJECT/$USER/frozen/fleurs24-asr-test
export FLEURS24_ASR_SET=$PROJECT/$USER/frozen/fleurs24-asr-test
```

### 1.6. Smoke test

Run a few samples of every benchmark before leaving the watcher unattended.

**1. Link a checkpoint.** Only creates a symlink.

```bash
mkdir -p checkpoints/smurf
ln -s /e/project1/e-ext-2025e01-100/speech_stream/eurollm_speech_config_run/checkpoints/step=1000.ckpt checkpoints/smurf/
```

**2. Check that everything resolves.** Runs nothing:

```bash
python scripts/evaluate.py plan configs/eval/smurf.yaml --site jupiter -v
pytest tests/test_ckpt_watch.py -q
```

The `skip` lines say what is missing. librispeech, air-bench and mcif should show as `missing` (pending), not skipped.

**3. Run 5 samples per benchmark.**

```bash
python scripts/evaluate.py run configs/eval/smurf.yaml --site jupiter --limit 5 --wait
```

Output goes to `$EVAL_ROOT/smurf/smoke-limit5/`. If a job fails, read the `slurm-*.out` in its slice folder. While it runs, from another shell:

```bash
squeue --me -o "%.10i %.25j %.8T %.10M %.10l %R"
watch -n 30 'squeue --me -o "%.10i %.25j %.8T %.10M %R"'     # same, refreshing
sacct -j <id> --format=JobID,Elapsed,State                    # how long a job took
```

## 2. Run the watcher

`scripts/watch_checkpoints.py` polls the training run's checkpoint folder. For each new `step=N.ckpt` it waits until the file has been untouched for `settle_minutes`, decides whether `every_n` selects it, and if so submits the evaluation as SLURM jobs. When the jobs finish it rebuilds the report, one column per checkpoint. It never deletes a checkpoint.

```
new step=N.ckpt ─► written? (untouched for settle_minutes)
                 ─► every_n picks it? ─► configs/step-N.yaml ─► evaluate.py run --site ... ─► SLURM
results/summary.csv (one column per checkpoint) ◄─ report ◄─ done ◄─ jobs finished
candidates ─► list what retention does not keep; you delete
```

Settings are in [configs/watch/smurf.yaml](../configs/watch/smurf.yaml): `ckpt_dir`, `pattern`, `eval_config`, `every_n` (1 = every checkpoint; 2 = the 1st, 3rd, 5th found, so the rest show as `skipped`), `max_in_flight` and `retention`.

### 2.1. Launch

```bash
tmux new -s watcher
python scripts/watch_checkpoints.py run configs/watch/smurf.yaml
```

Options: `--once` does a single pass and exits (usable from cron or a job); `--interval` sets the seconds between passes (default 300).

### 2.2. Check it is alive

```bash
pgrep -af watch_checkpoints        # must print a line
```

Iif `tmux attach` says `no sessions`, the process is gone. Start it again: it resumes from the ledger (`$EVAL_ROOT/smurf-run/watch_state.json`), loses nothing and never evaluates a step twice.

### 2.3. Baselines

`baselines_models` in the watch config lists fixed models that are evaluated **once**, on the same benchmarks as the checkpoints, into the same `output_dir`. Their logs stay in `logs/` and the report gets one extra column per baseline. They do not count in `max_in_flight` or retention.

- Download their weights first (1.4, rows 6-7).
- A `failed` baseline is not retried by itself. After fixing the cause: `python scripts/watch_checkpoints.py rerun-baselines configs/watch/smurf.yaml` (slices that already succeeded are left alone).
- To run without baselines,comment out these two keys in [configs/watch/smurf.yaml](../configs/watch/smurf.yaml) and restart the watcher:

  ```yaml
  baselines_config: configs/eval/baselines-v1.yaml
  baselines_models: [qwen2-audio-7b-instruct, qwen3-omni-30b-a3b-instruct]
  ```

  Logs already written stay in `logs/` and keep showing in the report.

### 2.4. Some things to take into account

- **Ledger**: delete an entry (or use `enqueue`) to redo a step.
- **`every_n`** counts checkpoints in the order they were discovered (the 1st, then every n-th). Use `enqueue` for the last checkpoint of a run.
- **Retention** keeps: the newest checkpoint (training resumes from it), the last `keep_last`, the `keep_best.k` best by `metric` (`benchmark/slice/metric`, as in `results/results_long.csv`), every multiple of `keep_every`, and anything not finished evaluating or failed. The rest is only listed by `candidates`; the script never deletes.

## 3. Follow progress and get results

### 3.1. Status

```bash
python scripts/watch_checkpoints.py status configs/watch/smurf.yaml   # step, status, metric, keep or candidate
squeue --me
```

`status` shows `done`, `skipped` (not picked by `every_n`), `failed` (see the `slurm-*.out` of the slice folder) or still running. The `metric` column is the one `retention.keep_best.metric` points at.

### 3.2. Results

Everything is under `$EVAL_ROOT/smurf-run/`:

| Path                                        | What it holds                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| `results/summary.csv`                     | headline metric per benchmark/slice, one column per checkpoint                  |
| `results/results_long.csv`                | every metric, one row each (the names`keep_best.metric` uses)                 |
| `results/results.xlsx`                    | summary, one sheet per benchmark, and`runs` (which log each number came from) |
| `logs/<model>/<benchmark>/<slice>/*.json` | inspect logs: every sample, prompt, answer and score                            |
| `logs/.../slurm-<jobid>.out`              | stdout of each job                                                              |

```bash
cd $EVAL_ROOT/smurf-run
column -s, -t < results/summary.csv | less -S
grep -i mcif results/results_long.csv                  # find the exact metric name for retention
inspect view --log-dir logs                            # browse samples
python $PROJECT/$USER/eval/scripts/evaluate.py report configs/*.yaml   # rebuild results/ by hand
```

### 3.3. Extra commands

These commands are extra and run outside the normal loop.

**Evaluate a specific checkpoint.** The watcher skips checkpoints according to `every_n`, so some steps show as `skipped` in `status` (for example the last one of a run). To evaluate one anyway:

```bash
python scripts/watch_checkpoints.py enqueue configs/watch/smurf.yaml 8000     # 8000 = the step number
```

This does not run anything by itself: it marks the step as pending in the ledger, and the **watcher must be running** to submit it on its next pass. It also works to retry a `failed` step once the cause is fixed.

**See which checkpoints can be deleted.** Checkpoints are large, and retention decides which ones to keep (see 2.3). To list the rest:

```bash
python scripts/watch_checkpoints.py candidates configs/watch/smurf.yaml
```

This deletes nothing. It prints the candidates and writes one `rm` line per checkpoint to `$EVAL_ROOT/smurf-run/delete-candidates.txt`.
