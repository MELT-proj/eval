# speechval-dev

The dev half of the speechval validation/test partition: full `validation`/`dev` splits, except where noted (VoxPopuli and CoVoST2 are capped, see below),
listed in `configs/eval/suites/speechval-dev.yaml`.

| benchmark | spec | source | dataset_ids | env var |
|---|---|---|---|---|
| LibriSpeech | `configs/hf/librispeech-hf-validation-{clean,other}.yaml` | HF, read live | `librispeech-validation-{clean,other}` | - |
| VoxPopuli | `configs/shar/voxpopuli-dev.yaml` | frozen | `voxpopuli-<lang>-dev` (14) | `VOXPOPULI_DEV_SET` |
| FLEURS X->en | `configs/shar/fleurs24-st-xen-dev-full.yaml` | frozen | `fleurs-<lang>_en-dev` (23) | `FLEURS24_ST_XEN_DEV_SET` |
| FLEURS en->X | `configs/shar/fleurs24-st-enx-dev.yaml` | frozen | `fleurs-en_<lang>-dev` (23) | `FLEURS24_ST_ENX_DEV_SET` |
| CoVoST2 | `configs/shar/st-eval-campaign-v1-dev.yaml` | frozen | `covost2-<src>_en-dev` (21), `covost2-en_<tgt>-dev` (15) | `COVOST2_DEV_SET` |

Not part of it: `fleurs24-asr-dev.yaml` (FLEURS ASR), and `fleurs24-st-xen-dev.yaml` (training's 100-per-language
subsample; the full twin above is separate).

Every benchmark covers all of its languages; only dev is capped, per language or direction (VoxPopuli 1000, CoVoST2 2000).
VoxPopuli dev has 14 languages, not 16: et (47 usable cuts) and lt (3) are too small to score.
CoVoST2 has 36 directions: 21 X->en (ar ca cy de es et fa fr id it ja lv mn nl pt ru sl sv ta tr zh) and 15 en->X
(ar ca cy de et fa id ja lv mn sl sv ta tr zh). Suites: `covost2-dev` / `covost2-enx-dev`.

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
  are plain Shar without `.idx`. The 6 missing FLEURS validation splits were copied from the nyx tree (indexed already)
  into `/e/data1/smurf4eu-data/speech/shar/fleurs/<locale>/validation` and symlinked into the indexed dev tree below. The existing indexed tree (`.../calvo-bartolome1/azeros-indexed`) holds only `test`,
  so the validation splits were indexed into `/e/data1/smurf4eu-data/speech/azeros-indexed` (symlinks + `.idx`, small).
  There was no CoVoST2 on jupiter: `covost2/<dir>/{dev,test}` for all 36 directions (~56 GB, indexed already) and the
  `test` splits of the 6 FLEURS locales above were copied from the nyx tree (`rsync -rt`) into
  `/e/data1/smurf4eu-data/speech/shar/{covost2,fleurs}/`.
* The indexed tree used by the freeze is `/e/data1/smurf4eu-data/speech/azeros-indexed` (on data1, not scratch):
  `fleurs/<24 locales>/{validation,test}` and `voxpopuli/<14>/validation`; the 6 copied locales are symlinks to the shar tree.
* **nyx/artemis**: `/mnt/scratch-nyx/giuseppe/melt/melt-data/shar` is already indexed and has `covost2/*_en/dev`.

```bash
# jupiter (venv with lhotse: $PROJECT/calvo-bartolome1/venvs/freeze-eval; code from this repo via PYTHONPATH)
export AZEROS_DIR=/e/data1/smurf4eu-data/speech/azeros AZEROS_INDEXED_DIR=/e/data1/smurf4eu-data/speech/azeros-indexed
python scripts/index_shar.py $AZEROS_DIR $AZEROS_INDEXED_DIR \
    fleurs/{en_us,es_419,fr_fr,de_de,it_it,nl_nl,pt_br,hu_hu,fi_fi,cs_cz,hr_hr,bg_bg,lv_lv,da_dk,lt_lt,et_ee,ga_ie,mt_mt}/validation  # + el_gr pl_pl ro_ro sk_sk sl_si sv_se: copied from nyx, see above \
    voxpopuli/{en,de,fr,es,pl,it,ro,hu,cs,nl,fi,hr,sk,sl}/validation
export LOCAL_DATASETS_DIR=$AZEROS_INDEXED_DIR F=/e/data1/smurf4eu-data/speech/frozen-dev
for s in fleurs24-st-xen-dev-full fleurs24-st-enx-dev voxpopuli-dev; do melteval freeze configs/shar/$s.yaml -o $F/$s; done

# jupiter, CoVoST2 (copied tree, see above)
LOCAL_DATASETS_DIR=/e/data1/smurf4eu-data/speech/shar melteval freeze configs/shar/st-eval-campaign-v1-dev.yaml -o $F/st-eval-campaign-v1-dev

# nyx / artemis (same set, for runs there)
LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar \
  melteval freeze configs/shar/st-eval-campaign-v1-dev.yaml -o /mnt/scratch-artemis/giuseppe/frozen/speechval-dev/st-eval-campaign-v1-dev
```

All jupiter frozen sets live in `/e/data1/smurf4eu-data/speech/frozen-dev` (re-frozen on 2026-11-10 from the data1 indexed
tree, with all 36 CoVoST2 directions; the previous sets, which pointed at a scratch tree, are in `frozen-dev-superseded`).
Counts: VoxPopuli 13034, FLEURS X->en 8261, FLEURS en->X 8819, CoVoST2 61452.

The test counterparts of the two sets that changed are in `/e/data1/smurf4eu-data/speech/frozen-test`:
`covost2-test` (36 directions, 329103 samples; env `COVOST2_SET`) and `fleurs24-st-xen-test` (23 directions, 18816;
env `FLEURS24_ST_XEN_SET`). VoxPopuli and FLEURS en->X test were already complete.

Frozen sets (locators are absolute paths, so they only work on their own machine):

| env var | path |
|---|---|
| `VOXPOPULI_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/voxpopuli-dev` |
| `FLEURS24_ST_XEN_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/fleurs24-st-xen-dev-full` (23 directions) |
| `FLEURS24_ST_ENX_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/fleurs24-st-enx-dev` |
| `COVOST2_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/st-eval-campaign-v1-dev`; nyx/artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/st-eval-campaign-v1-dev` |

Samples (capped sets, see below): VoxPopuli 13034 (1000 each for en, de, fr, es, pl, it, ro, hu, cs, nl; sl 889,
hr 789, fi 687, sk 669). The freezer drops cuts with empty `text` (VoxPopuli validation has untranscribed cuts: hr
loses 496 of 1285). FLEURS X->en 8261 (23 directions, source text never empty); FLEURS en->X 8819 (23 directions; see `dropped_no_reference`
per source, expected); CoVoST2 10000 (2000 per direction: de, es, fr, it, pt). The CoVoST2 jupiter and nyx/artemis
sets are identical in size.

## Caps

Running the full dev splits (CoVoST2 53.7k samples, VoxPopuli 18k) takes too long, so those two are capped with
`max_samples` in the spec; the freezer's seeded shuffle picks the samples, so a cap is reproducible. FLEURS and
LibriSpeech are run in full.

* VoxPopuli: 1000 per language. Full frozen counts per language (after untranscribed cuts are dropped): de 2109,
  en 1753, fr 1727, pl 1662, es 1631, ro 1371, it 1257, nl 1229, cs 1095, hu 1054, sl 889, hr 789, fi 687, sk 669.
  Languages under 1000 keep everything. et (47) and lt (3) are not in the dev set: too small to score.
* CoVoST2: 2000 per direction (10,000 in total, from 53,692).

The uncapped sets are kept next to the capped ones, renamed `*.full` (CoVoST2 on both machines, VoxPopuli on jupiter;
the uncapped VoxPopuli set still includes et and lt).
