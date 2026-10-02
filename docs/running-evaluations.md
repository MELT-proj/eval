# Running evaluations: models × benchmarks

One command evaluates a list of models on a list of benchmarks and puts every result in one folder:

```bash
python scripts/evaluate.py run configs/eval/baselines-v1.yaml --site bocconi --wait
```

## 1. Naming a model

Models are named like any inspect model, `<provider>/<model>`. The provider is
who runs it:

| `model:`              | What runs                                       | Prompt path (`-T prompt_style`) |
| ----------------------- | ----------------------------------------------- | --------------------------------- |
| `melt/<checkpoint>`   | a MELT checkpoint, MELT's`generate()`         | `melt`                          |
| `melt/hf/qwen2_audio` | Qwen2-Audio-7B-Instruct, transformers           | `hf`                            |
| `melt/hf/qwen3_omni`  | Qwen3-Omni-30B-A3B-Instruct, transformers       | `hf`                            |
| `smurf/<checkpoint>`  | a SMURF (NeMo) checkpoint, SALM's`generate()` | `smurf`                         |
| `melt/vllm/<model>`   | reserved, not implemented                       | —                                |

`melt/hf/<model>` loads a Hub revision pinned in the code (the table in [baseline-providers.md](baseline-providers.md)); there is no path to pass.
The prompt path follows from the name, and the launcher sets it.

## 2. Models x benchmarks config

One YAML per campaign, e.g. [configs/eval/baselines-v1.yaml](../configs/eval/baselines-v1.yaml):

```yaml
name: baselines-v1
output_dir: ${EVAL_ROOT}/baselines-v1      # where everything goes

models:
  - name: qwen2-audio                      # folder name and row in the report
    model: melt/hf/qwen2_audio
    args: [-M, batch_size=8]               # extra `inspect eval` arguments
    skip: [mcif-long-*]                    # dataset_ids not to run
    instruction:                           # for benchmarks with `needs_instruction`
      asr: "Transcribe this audio."
  - name: my-melt-run
    model: melt/checkpoints/melt/MA-v1.2.7 # relative = from the repo root
    venv: ${MELT_VENV_PATH}                # optional: a venv other than the default

benchmarks:
  - name: librispeech
    spec: configs/librispeech-hf-test-clean.yaml   # or  frozen_set: <dir>
    needs_instruction: true                # its samples carry no prompt of their own
    slices:                                # one job per slice
      - {task: asr, dataset_id: librispeech-test-clean, time: "02:00:00"}
  - name: air-bench
    spec: configs/hf/air-bench.yaml
    judge: melt/hf/qwen3_omni              # grades the audio_chat slices
    slices:
      - {task: audio_mcq, dataset_id: air-bench-foundation-speech}
      - {task: audio_chat, dataset_id: air-bench-chat-speech}
```

- A **slice** is one `(task, dataset_id)` of a benchmark. A **job** is one
  model on one slice:, i.one `inspect eval`, one log.
- `instruction` is only used for `hf` and `smurf` models. MELT reads its prompt
  from the checkpoint's `training_config.yaml`.
- A slice may add `args` (e.g. `[--max-tokens, "4096"]`) and `time` (the SLURM
  wall clock).

## 3. The commands

```bash
python scripts/evaluate.py plan     CONFIG --site bocconi   # what would run, what is done, what is skipped and why
python scripts/evaluate.py run      CONFIG --site bocconi   # start every job without a successful log
python scripts/evaluate.py status   CONFIG --site bocconi   # done / queued / failed / missing
python scripts/evaluate.py score    CONFIG --site bocconi   # grade audio_chat slices with the benchmark's judge
python scripts/evaluate.py report   CONFIG --site bocconi   # CSV + Excel
python scripts/evaluate.py prefetch CONFIG --site bocconi   # download datasets + baseline weights (login node)
```

The usual loop:

1. `prefetch` once. Compute nodes may have no internet.
2. `plan`, then read it.
3. `run --wait`: submits, waits for the queue to empty, writes the report.
4. `status`. If something failed, fix it and `run` again: only what has no
   successful log is started again.
5. `score --wait`, if a benchmark has a judge.

Useful options: `--models a b`, `--benchmarks x`, `--out DIR` (instead of
`output_dir`), and `run --limit 5` for a smoke test (5 samples per job, written
to `<output_dir>/smoke-limit5/` so it never counts as the real run).

### Where it runs

- `--site <name>`: through SLURM, using `infra/sites/<name>.sh` for the venv,
  partition and paths. That file is the only per-machine piece: to add a
  machine, copy `infra/sites/example.sh`.
- `--local`: one job after another on this machine, in the current venv. Use
  it on a machine where the GPU is yours or inside an `srun` allocation, never
  on a shared GPU outside SLURM (see AGENTS.md).

### What calls what

```
scripts/evaluate.py run
  └─ per job:  infra/runners/submit_eval.sh <site> <model> <eval_set> <args>   (SLURM)
               or bash infra/run_eval.sbatch  <model> <eval_set> <args>         (--local)
                 └─ inspect eval melteval/tasks.py@speech --model <model> -T prompt_style=<family> ...
                      └─ melteval/providers/router.py → MELTAPI | Qwen2AudioAPI | Qwen3OmniAPI
                         (smurf/... → SmurfAPI)
```

`run_eval.sbatch` also works on its own, for a single job:
`infra/runners/submit_eval.sh bocconi melt/hf/qwen2_audio configs/librispeech-hf-test-clean.yaml -T task_filter=asr -T instruction='"Transcribe this audio."'`.

## 4. The judge

`score` grades the `audio_chat` slices of a benchmark with `judge:`, which is any model name:

```yaml
judge: melt/hf/qwen3_omni        # local: loaded text-only; a GPU job (SLURM, or here with --local)
judge: openai/gpt-4o             # a paid API: needs OPENAI_API_KEY; run here, no GPU
judge_args: {temperature: 0}     # optional model arguments for the judge
judge_venv: ${JUDGE_VENV}        # optional venv (e.g. one with the provider's SDK)
```

A log already graded by the judge's scorer is skipped (`--rescore` to redo).
The scores are written into the same log, so `report` picks them up.

## 5. The output folder

The config's `output_dir`, usually `${EVAL_ROOT}/<campaign>`.

To put it somewhere else:

- one command: `--out /another/path`;
- one campaign: change `output_dir:` in its YAML;
- every campaign on a machine: `export EVAL_ROOT=/another/path` in the shell,
  or change the default in `infra/sites/<site>.sh`.

```
<output_dir>/
  config.yaml                              copy of the config used
  submissions.jsonl                        one line per job started: when, how, SLURM job id
  logs/<model>/<benchmark>/<slice>/
      <timestamp>_speech-....json          the inspect log: every sample, prompt, answer, score
      slurm-<jobid>.out                    the job's stdout (local runs: local-<time>.out)
  results/
      summary.csv                          headline metric per (benchmark, slice) x model
      results_long.csv                     every metric, one row each
      results.xlsx                         summary, one sheet per benchmark, and `runs`
```

- A job is done when its folder holds a log with `status: success`. That is
  all `run` and `status` look at, so deleting a log re-runs the job.
- `inspect view --log-dir <output_dir>/logs` browses the logs sample by sample.
- The `runs` sheet names the log each number came from. When a slice has
  several successful logs, the newest wins.
