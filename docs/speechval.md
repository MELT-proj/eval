# speechval-dev

The dev half of the speechval validation/test partition: full `validation`/`dev` splits (no `max_samples`),
listed in `configs/eval/suites/speechval-dev.yaml`.

| benchmark | spec | source | dataset_ids | env var |
|---|---|---|---|---|
| LibriSpeech | `configs/hf/librispeech-hf-validation-{clean,other}.yaml` | HF, read live | `librispeech-validation-{clean,other}` | - |
| VoxPopuli | `configs/shar/voxpopuli-dev.yaml` | frozen | `voxpopuli-<lang>-dev` (16) | `VOXPOPULI_DEV_SET` |
| FLEURS X->en | `configs/shar/fleurs24-st-xen-dev-full.yaml` | frozen | `fleurs-<lang>_en-dev` (23) | `FLEURS24_ST_XEN_DEV_SET` |
| FLEURS en->X | `configs/shar/fleurs24-st-enx-dev.yaml` | frozen | `fleurs-en_<lang>-dev` (23) | `FLEURS24_ST_ENX_DEV_SET` |
| CoVoST2 | `configs/shar/st-eval-campaign-v1-dev.yaml` | frozen | `covost2-<src>_en-dev` (de/es/fr/it/pt) | `COVOST2_DEV_SET` |

Not part of it: `fleurs24-asr-dev.yaml` (FLEURS ASR), and `fleurs24-st-xen-dev.yaml` (training's 100-per-language
subsample; the full twin above is separate). The CoVoST2 test spec also has `en_de`, which no suite runs, so it has no
dev twin.

## FLEURS en->X references

`configs/refs/fleurs-<lang>-reference-dev.json` (23 targets) come from FLEURS `dev.tsv` at the pinned revision of
`google/fleurs`. Needs internet (mollete/nyx), already committed:

```bash
python scripts/build_fleurs_reference.py bg_bg cs_cz da_dk de_de el_gr es_419 et_ee fi_fi fr_fr ga_ie hr_hr hu_hu \
    it_it lt_lt lv_lv mt_mt nl_nl pl_pl pt_br ro_ro sk_sk sl_si sv_se --splits validation
```

A target lacks some English sentences (nl has 117 of 150 ids), so en->X sets are smaller than the audio;
the difference is `dropped_no_reference` in `frozen_set.json`.

## LibriSpeech prefetch (login shell; compute nodes are offline)

```bash
export HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache   # adjust on the cluster you run on
python - <<'PY'
from datasets import load_dataset
for name in ("clean", "other"):
    load_dataset("openslr/librispeech_asr", name, split="validation", revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1")
PY
```

The pinned revision has `clean/validation` and `other/validation` parquet files (checked).

## Freezing (CPU only)

What was run (2026-11-06). `$LOCAL_DATASETS_DIR` is not set in a jupiter login shell; the Shar trees are:

* **jupiter**: raw `AZEROS_DIR=/e/data1/smurf4eu-data/speech/azeros` has `voxpopuli/<lang>/validation` for all 16
  languages and `fleurs/<locale>/validation` for 18 locales (no el_gr, pl_pl, ro_ro, sk_sk, sl_si, sv_se), but both
  are plain Shar without `.idx`. The existing indexed tree (`.../calvo-bartolome1/azeros-indexed`) holds only `test`,
  so the validation splits were indexed into `$SCRATCH/attanasio1/azeros-indexed-dev` (symlinks + `.idx`, small).
  There is **no CoVoST2 on jupiter**.
* **nyx/artemis**: `/mnt/scratch-nyx/giuseppe/melt/melt-data/shar` is already indexed and has `covost2/*_en/dev`.

```bash
# jupiter (venv with lhotse: $PROJECT/calvo-bartolome1/venvs/freeze-eval; code from this repo via PYTHONPATH)
export AZEROS_DIR=/e/data1/smurf4eu-data/speech/azeros AZEROS_INDEXED_DIR=$SCRATCH/attanasio1/azeros-indexed-dev
python scripts/index_shar.py $AZEROS_DIR $AZEROS_INDEXED_DIR \
    fleurs/{en_us,es_419,fr_fr,de_de,it_it,nl_nl,pt_br,hu_hu,fi_fi,cs_cz,hr_hr,bg_bg,lv_lv,da_dk,lt_lt,et_ee,ga_ie,mt_mt}/validation \
    voxpopuli/{en,de,fr,es,pl,it,ro,hu,cs,nl,fi,hr,sk,sl,et,lt}/validation
export LOCAL_DATASETS_DIR=$AZEROS_INDEXED_DIR F=$PROJECT/attanasio1/frozen-dev
for s in fleurs24-st-xen-dev-full fleurs24-st-enx-dev voxpopuli-dev; do melteval freeze configs/shar/$s.yaml -o $F/$s; done

# nyx / artemis
LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar \
  melteval freeze configs/shar/st-eval-campaign-v1-dev.yaml -o /mnt/scratch-artemis/giuseppe/frozen/speechval-dev/st-eval-campaign-v1-dev
```

Frozen sets (locators are absolute paths, so they only work on their own machine):

| env var | path |
|---|---|
| `VOXPOPULI_DEV_SET` | jupiter `/e/project1/e-ext-2025e01-100/attanasio1/frozen-dev/voxpopuli-dev` |
| `FLEURS24_ST_XEN_DEV_SET` | jupiter `.../frozen-dev/fleurs24-st-xen-dev-full` (17 directions) |
| `FLEURS24_ST_ENX_DEV_SET` | jupiter `.../frozen-dev/fleurs24-st-enx-dev` |
| `COVOST2_DEV_SET` | nyx/artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/st-eval-campaign-v1-dev` |

Samples: VoxPopuli 17972 (en 1753, de 2109, fr 1727, es 1631, pl 1662, it 1257, ro 1371, hu 1054, cs 1095, nl 1229,
fi 687, hr 789, sk 669, sl 889, et 47, lt 3); the freezer drops cuts with empty `text` (VoxPopuli validation has
untranscribed cuts: hr loses 496 of 1285). FLEURS X->en 6234 (17 directions); FLEURS en->X 8819 (23 directions;
see `dropped_no_reference` per source, expected); CoVoST2 53692 (de 13474, es 13221,
fr 14749, it 8930, pt 3318). A CoVoST2 run on jupiter needs the tree copied there first.
