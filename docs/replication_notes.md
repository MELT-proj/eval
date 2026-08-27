# Testing smurf branch integration

# 1. Create venv compatible with smurf code (speechllm)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ../smurf/speechllm --override smurf_overrides.txt
uv pip install -e ".[hf,metrics]
```

## 2. Logic ok (pytest)

```Shell
pytest tests -q
```

# 3. Construct frozen set for evaluation

## 3.0. Generic config

```YAML
name: <frozen_set_name>
seed: 0

input_cfg:
  - type: hf_dataset
    repo: <usuario/dataset>       # HF repo
    revision: <commit-sha-o-tag>  # pinning
    split: test                   # optional, default "test"
    # name: <config-del-dataset>  # optional: el argumento "config" de HF (algunos datasets tienen varios)
    audio_column: audio           # optional, default "audio"
    text_column: text             # optional, default "text"
    # id_column: id               # optional: columna a usar como ID de la muestra
    # source_text_column: ...     # optional: solo para ST, el texto fuente (para COMET luego)
    # min_duration: 1.0           # optional: filtro de duración en segundos
    # max_duration: 30.0
    # max_samples: 50              # opcional: recorta el nº de muestras (subsampleo aleatorio con `seed`)
    tags:
      task: asr                   # asr | st | ...
      lang: en                    # idioma de la salida esperada
      # src_lang: en               # solo tareas con idioma fuente/destino (p.ej. st)
      # tgt_lang: de
      dataset_id: <etiqueta-libre> # para desgloses por corpus en el summary
```

## 3.1. Librispeech

* Config at [configs/librispeech-hf-smoke.yaml](configs/librispeech-hf-smoke.yaml)
* Build the frozen set

```Shell
melteval freeze configs/librispeech-hf-smoke.yaml -o runs/librispeech-hf-smoke
```

* Check it was constructed ok

```Shell
melteval show runs/librispeech-hf-smoke
```

# 4. Evaluate on the frozen set

## 3.1. Librispeech

```Shell
TMPDIR=~/.cache/triton-eval TRITON_CACHE_DIR=~/.cache/triton-eval \
inspect eval melteval/tasks.py@asr \
  --model smurf/data/checkpoints/step=2784-last.ckpt \
  -T frozen_set="$(pwd)/runs/librispeech-hf-smoke" \
  -T prompt_style=smurf \
  -T instruction='"Transcribe this English audio: "' \
  -T normalizer=none \
  -M batch_size=4 -M device_map=auto
```
