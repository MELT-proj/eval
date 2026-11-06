# Evaluating SMURF checkpoints on JUPITER

The watcher is a thin loop over `scripts/evaluate.py`: for each checkpoint it writes one config and calls `evaluate.py run --site jupiter`, so jobs are submitted with the settings in [infra/sites/jupiter.sh](../infra/sites/jupiter.sh). Jobs, logs and reports are the ones described in [running-evaluations.md](running-evaluations.md).

JUPITER only hands out whole nodes (4 GH200), so the slices are not submitted one 1-GPU job each but in **packs**: one SLURM job per node, one worker per GPU, the checkpoint loaded once per worker. A SMURF checkpoint (55 slices, eval suite v1) is 3 packs of at most 20 slices (`pack:` in [configs/eval/smurf-v1.yaml](../configs/eval/smurf-v1.yaml)) instead of 55 jobs, and the checkpoint is loaded 12 times instead of 55. The site file turns this on (`MELT_PACK_GPUS=4`). Details: [running-evaluations.md, "Packs"](running-evaluations.md#packs-whole-node-sites-jupiter).

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

**Metrics venv.** MCIF's official scorers (BERTScore for SQA, COMET for MCIF ST) and COMET for FLEURS run *after* generation, from a venv of their own (`$METRICS_VENV_PATH` in [infra/sites/jupiter.sh](../infra/sites/jupiter.sh)): `mcif-bench` and `unbabel-comet` pin their own `transformers`.

`mcif-bench` on PyPI is still the 1.x line, whose translation scorer drives RWTH's `mwerSegmenter` binary (x86-only). The 2.0 line replaced it with `mweralign`, which resegments in-process, so MCIF ST COMET runs here like anywhere else. Build the venv with the repo's script, which pins the 2.0 commit:

```bash
infra/setup/metrics_venv.sh $PROJECT/$USER/venvs/metrics-eval   # or: METRICS_VENV_PATH=... infra/setup/metrics_venv.sh
```

It is the equivalent of

```bash
uv venv --python 3.12 --python-preference only-managed $PROJECT/$USER/venvs/metrics-eval
VIRTUAL_ENV=$PROJECT/$USER/venvs/metrics-eval uv pip install -e . \
    "mcif-bench @ git+https://github.com/hlt-mt/mcif.git@cdb1d62a00281a14bef52f829b113fc73981c85d" \
    "setuptools<81" "transformers<5"
```

followed by an import check that would have caught the JUPITER failure below (`score_st`, `mweralign`, `comet`, `melteval`, and `httpcore`).

**Point `$METRICS_VENV_PATH` at that venv.** It is what the suite's `post_venv:` resolves at plan time, and it may already be exported in your shell to the older shared venv (built before the 2.0 upgrade, `mcif 1.0`, no `mweralign`). Check the one the jobs will actually use, not the one you just built:

```bash
"${METRICS_VENV_PATH%/bin/activate}/bin/python" -c "import mcif, mweralign; print(mcif.__version__)"   # must print 2.0
```

With 1.x the ST COMET job fails with `... mwerSegmenter is not in PATH and no MWERSEGMENTER_ROOT ...` — a loud error, but only after the job is queued and started.

Two aarch64 details: `mweralign` publishes x86_64 wheels only, so here `uv` builds it from its sdist (needs the login node's `gcc`/`g++`, which is enough; nothing to set up by hand). And `--python-preference only-managed` keeps the venv's interpreter under `$PROJECT`, which the compute nodes mount — a venv built against a `$HOME` interpreter only works on the login node.

Do **not** pass `--prerelease=allow`: it resolves `httpx` to its `1.0.dev*` line, which dropped `httpcore`, and every scoring job then dies on `import inspect_ai` with `ModuleNotFoundError: No module named 'httpcore'` before it reads a log. That is exactly how the first MCIF ST scoring attempt here failed (`metrics-mcif` venv) — all three jobs, in about a minute, with `mcif-bench`/`mweralign` themselves installed fine.

**Offline models for the post-hoc scorers.** The following need to be fetched in `$HF_HOME` from the login node: `Unbabel/wmt22-comet-da` and `xlm-roberta-large` (COMET), `roberta-large` (BERTScore, en) and `bert-base-multilingual-cased` (BERTScore, de/it), `google/metricx-24-hybrid-xxl-v2p6-bfloat16` and `google/mt5-xl` (MetricX, ST; the revisions pinned in `melteval/rescore.py`).

**MCIF references are a download too.** `mcif_scoring` re-fetches `MCIF.<track>.<lang>.ref.xml.gz` itself (it scores against the exact revision generation used, not whatever the dataset cache happens to hold), and the compute nodes are offline. Cache the two files the ST slices need from the login node — the keys come from the logs, so this works for logs someone else generated:

```bash
export HF_HUB_OFFLINE=0
python - <<'EOF'
import glob, json
from melteval.readers.mcif import download_reference_gz
keys = set()
for path in glob.glob(f"{EVAL_ROOT}/{CAMPAIGN}/logs/*/mcif/chunked_st-*/*"):
    if not path.endswith((".json", ".eval")):      # .out files live there too
        continue
    meta = json.load(open(path))["samples"][0]["metadata"]
    keys.add((meta["repo"], meta["revision"], meta["track"], meta["lang"]))
for key in sorted(keys):
    print("fetch", key, download_reference_gz(*key))
EOF
export HF_HUB_OFFLINE=1
```

(`evaluate.py prefetch` is not the tool here: it also downloads model snapshots, which scoring does not need.) If the logs were generated under another account's `HF_HOME`, symlink that account's `hub/<repo>` into yours read-only instead of downloading again, e.g. `ln -s /e/scratch/<project>/<user>/hf_cache/hub/datasets--FBK-MT--MCIF $HF_HOME/hub/`.

**Freezing needs the training package.** `melteval freeze` over a Shar source takes its reference rules from `melt-proj`, which the SMURF venv does not have. Freeze from a venv with `melt-eval[shar]` (the one that froze `fleurs24-st-xen-test`).

### 1.4. Downloads

The running nodes have no internet and jobs run with `HF_HUB_OFFLINE=1`, so anything missing from the cache fails the job. Download it here, into `$HF_HOME` (`$SCRATCH/$USER/hf_cache`).

| # | What                                        | Used for                                       |
| - | ------------------------------------------- | ---------------------------------------------- |
| 1 | `utter-project/EuroLLM-9B-Instruct-2512`  | the SMURF's LLM                                |
| 2 | `nvidia/canary-1b-v2`                     | the SMURF's audio encoder (`pretrained_asr`) |
| 3 | MCIF audio and references (`FBK-MT/MCIF`) | `mcif`                                       |
| 4 | LibriSpeech test-clean and test-other       | `librispeech`, `librispeech-other`           |
| 5 | `Qwen/Qwen2-Audio-7B-Instruct`            | baseline (2.3)                                 |
| 6 | `Qwen/Qwen3-Omni-30B-A3B-Instruct`        | baseline (2.3), about 60 GB                    |
| 7 | `roberta-large`, `bert-base-multilingual-cased`, `Unbabel/wmt22-comet-da`, `xlm-roberta-large` | BERTScore (MCIF SQA: `roberta-large` for en, the multilingual BERT for de/it) and COMET, post-hoc (2.5) |
| 8 | `openai/whisper-large-v3`                 | baseline (2.3)                                 |
| 9 | `DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B`, `DeSTA-ntu/Llama-3.1-8B-Instruct` (its `llm_model_id`, not `meta-llama/...`) | baseline (2.3); also needs its own venv, see [baseline-providers.md](baseline-providers.md) |

```bash
cd $PROJECT/$USER/eval
export HF_HOME=$SCRATCH/$USER/hf_cache HF_HUB_OFFLINE=0     # allow downloads in this shell

# 1-2. The SMURF's models
huggingface-cli download utter-project/EuroLLM-9B-Instruct-2512
huggingface-cli download nvidia/canary-1b-v2
ls $HF_HOME/hub/models--nvidia--canary-1b-v2/snapshots/*/    # must contain canary-1b-v2.nemo

# 4. LibriSpeech
python scripts/evaluate.py prefetch configs/eval/smurf-v1.yaml --benchmarks librispeech librispeech-other

# 7. Post-hoc metric models (the short names are the ones bert-score and COMET ask for)
huggingface-cli download roberta-large
huggingface-cli download bert-base-multilingual-cased
huggingface-cli download Unbabel/wmt22-comet-da
huggingface-cli download xlm-roberta-large
huggingface-cli download google/metricx-24-hybrid-xxl-v2p6-bfloat16
huggingface-cli download google/mt5-xl

# 5-6, 8-9. The baselines' models
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
huggingface-cli download openai/whisper-large-v3
huggingface-cli download DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B
huggingface-cli download DeSTA-ntu/Llama-3.1-8B-Instruct    # the backbone DeSTA's config.json names (llm_model_id)

# 3. MCIF, audio included
python - <<'EOF'
from melteval.dataset import spec_dataset
from melteval.readers.base import reader_for_locator
from melteval.manifest import AudioLocator
for dataset_id in ["mcif-short-fixed-en", "mcif-short-fixed-de", "mcif-short-fixed-it"]:
    ds = spec_dataset("configs/hf/mcif.yaml", dataset_id=dataset_id)   # ASR/ST chunks and SQA questions
    for s in ds.samples:
        loc = AudioLocator.from_dict(s.metadata["audio"]); reader_for_locator(loc).load_audio(loc)
    print(dataset_id, len(ds.samples), "ok")      # one "ok" line per dataset
EOF

export HF_HUB_OFFLINE=1
```

### 1.5. Local datasets

Three frozen sets, built once, from a venv with `melt-eval[shar]` (see 1.3).

| Set | Recipe | Variable | Source | Expected `dropped_no_reference` |
| --- | ------ | -------- | ------ | ------------------------------- |
| FLEURS ST EU→en | [configs/shar/fleurs24-st-xen-test.yaml](../configs/shar/fleurs24-st-xen-test.yaml) | `FLEURS24_ST_XEN_SET` | `$LOCAL_DATASETS_DIR/fleurs` | 0 |
| FLEURS ST en→EU | [configs/shar/fleurs24-st-enx-test.yaml](../configs/shar/fleurs24-st-enx-test.yaml) | `FLEURS24_ST_ENX_SET` | `$LOCAL_DATASETS_DIR/fleurs/en_us` | >0, on purpose: a target lacks some English sentences (nl: 251 of 350 kept) |
| VoxPopuli, 16 languages | [configs/shar/voxpopuli-test.yaml](../configs/shar/voxpopuli-test.yaml) | `VOXPOPULI_SET` | `$AZEROS_INDEXED_DIR/voxpopuli` | 0 |

VoxPopuli comes from the azeros tree (plain Shar, i.e., `cuts.*.jsonl.gz`, no `.idx`), so this conversion needs to be performed first:

```bash
F=$PROJECT/$USER/frozen
python scripts/index_shar.py $AZEROS_DIR $AZEROS_INDEXED_DIR \
    voxpopuli/{en,de,fr,es,pl,it,ro,hu,cs,nl,fi,hr,sk,sl,et,lt}/test

melteval freeze configs/shar/fleurs24-st-xen-test.yaml -o $F/fleurs24-st-xen-test
melteval freeze configs/shar/fleurs24-st-enx-test.yaml -o $F/fleurs24-st-enx-test
melteval freeze configs/shar/voxpopuli-test.yaml  -o $F/voxpopuli-test

export FLEURS24_ST_XEN_SET=$F/fleurs24-st-xen-test \
       FLEURS24_ST_ENX_SET=$F/fleurs24-st-enx-test VOXPOPULI_SET=$F/voxpopuli-test
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
python scripts/evaluate.py plan configs/eval/smurf-v1.yaml --site jupiter -v
pytest tests/test_ckpt_watch.py -q
```

The `skip` lines say what is missing. librispeech, librispeech-other and mcif should show as `missing` (pending), not skipped; fleurs24-* and voxpopuli are skipped until their variables are exported (1.5). The last lines list the packs `run` would submit: which slices go together, on 4 GPUs each.

**3. Run 5 samples per benchmark.**

```bash
python scripts/evaluate.py run configs/eval/smurf-v1.yaml --site jupiter --limit 5 --wait
```

This submits a few packs (one whole-node job each), not one job per slice. Output goes to `$EVAL_ROOT/smurf/smoke-limit5/`. Where to look:

- `packs/<id>-slurm-<jobid>.out`: the pack's own output, which slice each GPU started and finished (`[pack gpu2] done ...` / `FAILED ...`).
- `logs/<model>/<benchmark>/<slice>/pack-<jobid>-gpu<N>.out`: that slice's inspect output, including the traceback if it failed.
- `packs/<id>.json`: the manifest, which slices the pack holds.

## 2. Run the watcher

`scripts/watch_checkpoints.py` polls the training run's checkpoint folder. For each new `step=N.ckpt` it waits until the file has been untouched for `settle_minutes`, decides whether `every_n` selects it, and if so submits the evaluation as SLURM jobs (whole-node packs on JUPITER, see above). When the jobs finish it rebuilds the report, one column per checkpoint. It never deletes a checkpoint.

```
new step=N.ckpt ─► written? (untouched for settle_minutes)
                 ─► every_n picks it? ─► configs/<run>-step-N.yaml ─► evaluate.py run --site ... ─► SLURM (packs: 1 job per node, 1 worker per GPU)
results/summary.csv (one column per checkpoint) ◄─ report ◄─ scored ◄─ evaluate.py score ◄─ done ◄─ jobs finished
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

- Download their weights first (1.4, rows 5-6 and 8-9).
- DeSTA2.5 runs from its own venv (`DESTA_VENV_PATH`, install in [baseline-providers.md](baseline-providers.md)). `infra/sites/jupiter.sh` sets no default, so export it before starting the watcher; without it the DeSTA slices are skipped, with the reason.
- A `failed` baseline is not retried by itself. After fixing the cause: `python scripts/watch_checkpoints.py rerun-baselines configs/watch/smurf.yaml` (slices that already succeeded are left alone).
- To run without baselines,comment out these two keys in [configs/watch/smurf.yaml](../configs/watch/smurf.yaml) and restart the watcher:

  ```yaml
  baselines_config: configs/eval/baselines-v1.yaml
  baselines_models: [qwen2-audio-7b-instruct, qwen3-omni-30b-a3b-instruct, whisper-large-v3, desta2.5-audio]
  ```

  Logs already written stay in `logs/` and keep showing in the report.

### 2.4. Report names

Each checkpoint is a row named `<run>-step-<N>` (zero-padded, so a run's checkpoints sort by step). `<run>` is `run_name:` in the watch config, or else the name of `ckpt_dir` (of its parent when the folder is called `checkpoints`). The full path of the checkpoint is in every log (`melteval_checkpoint`) and in the `checkpoint` column of the `runs` sheet. Checkpoints already in a ledger from before this keep their old `step-<N>` name, so a running watcher keeps finding its logs.

### 2.5. Post-hoc scoring

Some metrics need a model of their own and are computed after generation, by `evaluate.py score`, from the metrics venv, in a SLURM job:

| Slices | Scorer | Metric in the report |
| ------ | ------ | -------------------- |
| MCIF SQA (`audio_chat`, short en/de/it) | `mcif_scoring.py@mcif_official_qa_scorer` | `official_bertscore` |
| MCIF QA + SUM (`audio_chat`, long en) | `mcif_scoring.py@mcif_official_qa_scorer` and `@mcif_official_sum_scorer` | `official_bertscore` on the QA and the summarisation samples of one log |
| MCIF ST (`chunked_st`, de/it) | `mcif_scoring.py@mcif_official_trans_scorer` | `official_trans_comet` |
| FLEURS and EuroparlST ST | `rescore.py@metricx_scorer` | `metricx` (MetricX-24 hybrid XXL error, 0-25, lower is better), and `metricx_coverage` (share of samples with an output) |

They are declared per slice in the suite, [configs/eval/suites/v1.yaml](../configs/eval/suites/v1.yaml) (`post_scorers:`, with `post_venv:` on the benchmark). The watcher runs them by itself: once a checkpoint's generation is `done`, each pass calls `evaluate.py score`, which submits what is missing and skips what is already queued, and the checkpoint becomes `scored` when nothing is left. Until then the report shows `pending` for those metrics. A missing score never makes a checkpoint `failed`; after 3 submissions that did not complete it, the watcher stops trying and `status` says so.

By hand, for any config:

```bash
python scripts/evaluate.py score $EVAL_ROOT/smurf-run/configs/<run>-step-0004000.yaml --site jupiter --wait
```

The official MCIF scorers need every sample of a `dataset_id` in the log: they cannot score a `--limit` run.

**Backfilling a metric on existing logs** (MCIF ST, before it ran on JUPITER) needs no regeneration — `score` only reads logs. Point a scoring-only config at the campaign, usually copying the one the watcher wrote and trimming it, and pass the campaign as `--out` so `plan` finds `logs/`:

```yaml
name: mcif-comet
output_dir: ${EVAL_ROOT}/mcif-comet       # or pass --out explicitly
skip: [air-bench, librispeech, librispeech-other, covost2, voxpopuli, fleurs24-asr,
       fleurs24-st-xen, fleurs24-st-enx, mcif-short-fixed-en, mcif-short-fixed-zh,
       mcif-short-mixed-*, mcif-long-*]
models:                                   # name = the logs/<name> folder, model = the checkpoint
  - name: <run>-step-0207000
    model: smurf/<absolute path to step=0207000.ckpt>
benchmarks: configs/eval/suites/v1.yaml
```

```bash
python scripts/evaluate.py score <that config> --out $EVAL_ROOT/mcif-comet --site jupiter --benchmarks mcif
```

`plan` skips a model whose checkpoint is not on this machine, and `score` inherits that: for logs copied from elsewhere, give the row any model of the right family (`model: melt/hf/qwen2_audio`) — only the folder name matters for scoring. Add `--models` to score one row, `--rescore` to redo a score that is already in the log (the default is to skip those). Scoring appends to the log in place and never regenerates samples; the copied logs need to be writable by whoever's job scores them.

Scoring is GPU work like generation: it goes through `infra/runners/submit_score.sh` → `infra/run_score.sbatch`, one SLURM job per (model, benchmark, scorer), and its stdout is `logs/<model>/mcif/score-slurm-<jobid>.out`.

### 2.6. Some things to take into account

- **Ledger**: delete an entry (or use `enqueue`) to redo a step.
- **Packs and time limits**: if a pack hits its wall clock, the slices it did not finish are `missing` and the next `run` (the watcher's next pass) puts them in a new pack. If that keeps happening, raise `pack: time` or lower `pack: max_slices` in `configs/eval/smurf-v1.yaml`.
- **`every_n`** counts checkpoints in the order they were discovered (the 1st, then every n-th). Use `enqueue` for the last checkpoint of a run.
- **Retention** keeps: the newest checkpoint (training resumes from it), the last `keep_last`, the `keep_best.k` best by `metric` (`benchmark/slice/metric`, as in `results/results_long.csv`), every multiple of `keep_every`, and anything not finished evaluating or failed. The rest is only listed by `candidates`; the script never deletes.

## 3. Follow progress and get results

### 3.1. Status

```bash
python scripts/watch_checkpoints.py status configs/watch/smurf.yaml   # step, status, metric, keep or candidate
squeue --me
```

`status` shows `done`, `skipped` (not picked by `every_n`), `failed` (see the `pack-*.out` of the slice folder) or still running, and whether a `done` checkpoint is `scored` (2.5). The `metric` column is the one `retention.keep_best.metric` points at.

### 3.2. Results

Everything is under `$EVAL_ROOT/smurf-run/`:

| Path                                        | What it holds                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------- |
| `results/summary.csv`                     | headline metric per benchmark/slice, one column per checkpoint                  |
| `results/results_long.csv`                | every metric, one row each (the names`keep_best.metric` uses)                 |
| `results/results.xlsx`                    | summary, one sheet per benchmark, and`runs` (which log each number came from) |
| `logs/<model>/<benchmark>/<slice>/*.json` | inspect logs: every sample, prompt, answer and score                            |
| `logs/.../pack-<jobid>-gpu<N>.out`        | inspect output of each slice run in a pack (`slurm-<jobid>.out` without packs)  |
| `packs/<id>.json`, `packs/<id>-slurm-<jobid>.out` | each pack's slices, and which GPU ran which (start/done/FAILED)        |
| `logs/.../score-slurm-<jobid>.out`        | stdout of each post-hoc scoring job (2.5)                                       |

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

**See which checkpoints can be deleted.** Checkpoints are large, and retention decides which ones to keep (see 2.6). To list the rest:

```bash
python scripts/watch_checkpoints.py candidates configs/watch/smurf.yaml
```

This deletes nothing. It prints the candidates and writes one `rm` line per checkpoint to `$EVAL_ROOT/smurf-run/delete-candidates.txt`.
