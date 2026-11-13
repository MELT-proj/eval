# speechval-dev

The dev half of the speechval validation/test partition: full `validation`/`dev` splits, except where noted (VoxPopuli and CoVoST2 are capped, see below),
listed in `configs/eval/suites/speechval-dev.yaml`.

| benchmark | spec | source | dataset_ids | env var |
|---|---|---|---|---|
| LibriSpeech | `configs/shar/librispeech-validation.yaml` | frozen | `librispeech-validation-{clean,other}` | `LIBRISPEECH_VALIDATION_SET` |
| VoxPopuli | `configs/shar/voxpopuli-dev.yaml` | frozen | `voxpopuli-<lang>-dev` (14) | `VOXPOPULI_DEV_SET` |
| FLEURS X->en | `configs/shar/fleurs24-st-xen-dev-full.yaml` | frozen | `fleurs-<lang>_en-dev` (23) | `FLEURS24_ST_XEN_DEV_SET` |
| FLEURS en->X | `configs/shar/fleurs24-st-enx-dev.yaml` | frozen | `fleurs-en_<lang>-dev` (23) | `FLEURS24_ST_ENX_DEV_SET` |
| CoVoST2 | `configs/shar/st-eval-campaign-v1-dev.yaml` | frozen | `covost2-<src>_en-dev` (21), `covost2-en_<tgt>-dev` (15) | `COVOST2_DEV_SET` |
| EuroparlST | `configs/shar/europarl-st-dev.yaml` | frozen | `europarl-st-<src>_<tgt>-dev` (72) | `EUROPARL_ST_DEV_SET` |

LibriSpeech is frozen rather than read live from the Hub: `datasets` prepares
every split of `openslr/librispeech_asr`'s `clean`/`other` config even when only
`validation` is requested (~60 GB of training audio), which offline compute
nodes cannot do. The audio and transcripts are the same; the frozen set is built
from the Lhotse `librispeech/{clean,other}/validation` tree.

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

## LibriSpeech validation (frozen)

`configs/shar/librispeech-validation.yaml` freezes `librispeech/{clean,other}/validation`
from the Lhotse tree (5567 samples: clean 2703, other 2864). Do not use the HF
`spec` for it: `datasets` prepares every split of `openslr/librispeech_asr`'s
`clean`/`other` config even when asked for `validation`, ~60 GB of training
audio, which offline compute nodes cannot fetch.

```bash
LOCAL_DATASETS_DIR=/path/to/shar \
  melteval freeze configs/shar/librispeech-validation.yaml -o <frozen>/librispeech-validation
```

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
Counts: VoxPopuli 13034, FLEURS X->en 8261, FLEURS en->X 8819, CoVoST2 61452 (36 directions). LibriSpeech validation (5567) is frozen per site from that site's Lhotse tree.

The test counterparts of the two sets that changed are in `/e/data1/smurf4eu-data/speech/frozen-test`:
`covost2-test` (36 directions, 329103 samples; env `COVOST2_SET`) and `fleurs24-st-xen-test` (23 directions, 18816;
env `FLEURS24_ST_XEN_SET`). VoxPopuli and FLEURS en->X test were already complete.

Frozen sets (locators are absolute paths, so they only work on their own machine):

| env var | path |
|---|---|
| `LIBRISPEECH_VALIDATION_SET` | artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/librispeech-validation`; jupiter: freeze `configs/shar/librispeech-validation.yaml` there and set this |
| `VOXPOPULI_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/voxpopuli-dev`; artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/voxpopuli-dev` |
| `FLEURS24_ST_XEN_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/fleurs24-st-xen-dev-full` (23 directions); artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/fleurs24-st-xen-dev-full` |
| `FLEURS24_ST_ENX_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/fleurs24-st-enx-dev`; artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/fleurs24-st-enx-dev` |
| `COVOST2_DEV_SET` | jupiter `/e/data1/smurf4eu-data/speech/frozen-dev/st-eval-campaign-v1-dev`; artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/st-eval-campaign-v1-dev` (36 directions, 61452) |
| `EUROPARL_ST_DEV_SET` | artemis `/mnt/scratch-artemis/giuseppe/frozen/speechval/europarl-st-dev` (72 directions, 36000) |

Samples (capped sets, see below): VoxPopuli 13034 (1000 each for en, de, fr, es, pl, it, ro, hu, cs, nl; sl 889,
hr 789, fi 687, sk 669). The freezer drops cuts with empty `text` (VoxPopuli validation has untranscribed cuts: hr
loses 496 of 1285). FLEURS X->en 8261 (23 directions, source text never empty); FLEURS en->X 8819 (23 directions; see `dropped_no_reference`
per source, expected); CoVoST2 61452 (36 directions, capped at 2000 each; the jupiter and artemis sets are identical in size).

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

## Canary-1B-v2 baseline (artemis, 2026-11-12)

First row of the suite: `configs/eval/speechval-canary.yaml`, output
`${EVAL_ROOT}/speechval-canary`. 92 supported slices (78 skipped: FLEURS `ga`,
10 non-EU CoVoST directions each way, 56 non-English Europarl pairs). MetricX
ran post-hoc for FLEURS and Europarl; CoVoST has the inline BLEU/chrF only, as
the suite declares.

Frozen sets used (artemis, under `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev`):

| set | spec_hash | samples | hours |
|---|---|---|---|
| librispeech-validation | `6cfe26369984` | 5567 | 10.5 |
| voxpopuli-dev | `6c8ac5cc07c7` | 13034 | 38.0 |
| fleurs24-st-xen-dev-full | `968f8151b34e` | 8261 | 25.9 |
| fleurs24-st-enx-dev | `e77d0be19482` | 8819 | 23.3 |
| st-eval-campaign-v1-dev | `cb0993207c8e` | 61452 | 94.3 |

Headline numbers (macro over the slices the model supports; WER is a mean of
per-slice corpus WER, not a pooled corpus figure):

| benchmark | metric | Canary-1B-v2 |
|---|---|---|
| LibriSpeech validation-clean | WER | 2.3 % |
| LibriSpeech validation-other | WER | 4.3 % |
| VoxPopuli dev (14 languages) | WER | 9.5 % |
| FLEURS X->en (22) | BLEU / MetricX | 28.0 / 5.14 |
| FLEURS en->X (22) | BLEU / MetricX | 28.9 / 4.58 |
| CoVoST2 X->en (11) | BLEU | 38.0 |
| CoVoST2 en->X (5) | BLEU | 30.1 |
| EuroparlST en<->X (16) | BLEU / MetricX | 32.2 / 4.12 |

## SMURF, Fondue and Cohere baselines (artemis, 2026-11-12/13)

Three more families on the same suite and frozen sets. Macro means over the slices
the model ran; WER is a mean of per-slice corpus WER; `BLEU/MetricX` are the
inline BLEU and the post-hoc MetricX-24 hybrid XXL score (lower is better).

| model | LibriSpeech | VoxPopuli | FLEURS X->en | FLEURS en->X | CoVoST X->en | CoVoST en->X | Europarl X->en | Europarl en->X |
|---|---|---|---|---|---|---|---|---|
| canary-1b-v2 | 3.3 % | 9.5 % | 28.0 / 5.14 | 28.9 / 4.58 | 38.0 / 5.43 | 30.1 / 5.62 | 34.6 / 4.58 | 29.9 / 3.67 |
| eurollm-speech-16n-step-600000 | 3.6 % | 11.8 % | 11.2 / 3.59 | 29.3 / 4.01 | 20.8 / 10.23 | 28.5 / 5.87 | 6.7 / 13.85 | 26.0 / 5.53 |
| fondue-step-3000 | 4.0 % | 11.0 % | 22.2 / 5.31 | 18.8 / 7.22 | 28.2 / 6.58 | 15.2 / 9.28 | 28.7 / 5.34 | 24.8 / 4.99 |
| fondue-step-17000 | 3.1 % | 9.3 % | 23.7 / 4.63 | 18.8 / 6.54 | 28.9 / 5.90 | 17.5 / 7.63 | 30.2 / 4.84 | 27.1 / 4.27 |
| fondue-step-20000 | 3.0 % | 9.4 % | 23.6 / 4.57 | 18.1 / 6.69 | 29.1 / 5.89 | 17.8 / 7.51 | 29.8 / 4.87 | 27.0 / 4.28 |
| cohere-transcribe-03-2026 | 1.9 % | 7.8 % | – | – | – | – | – | – |

Checkpoints and how they are loaded:

- **SMURF**: `eurollm_speech_16_nodes_run_1_asr_long/checkpoints/step=600000.ckpt`
  (21.8 GB), pulled from jupiter to
  `/mnt/scratch-nyx/giuseppe/checkpoints/smurf/...`. Loaded with
  `configs/smurf/eurollm_override_config.yaml` (`prompt_format: eurollm`;
  the stored value is the training default `qwen`) and the `smurf-eval` venv.
  CoVoST2 runs in the 16 en<->EU directions the model covers (MetricX now included).
- **Fondue**: `checkpoint-3000`, `checkpoint-17000` and `checkpoint-20000` (newest verified on 2026-11-13) from
  `/mnt/scratch-nyx/giuseppe/fondue-keep/` (HF-format MELT checkpoints,
  whisper-large-v3 + EuroLLM-1.7B). They were saved without a processor or
  `training_config.yaml`; both come from the run output
  (`/mnt/scratch-nyx/giuseppe/melt/outputs-fondue/Fondue/`). The MoE audio
  adapter needs the `main` training checkout, layered over the melteval venv by
  `pyextras/melt-main/activate`. Evaluation uses one fixed prompt per task
  (`configs/melt/fondue-eval-format.yaml`), not the run's random selection.
- **Cohere Transcribe**: `CohereLabs/cohere-transcribe-03-2026` (2B conformer
  encoder-decoder), pinned revision `d552dadc2922`,
  `melt/hf/cohere_transcribe` (see `docs/baseline-providers.md`). ASR only,
  so the five ST benchmarks are skipped, and of VoxPopuli's 14 slices only
  the seven languages it covers ran (`en de es fr it nl pl`): nine slices in
  all. The language is the instruction (`asr: "{lang_code}"`) because the
  processor takes one language per call and the model has no language
  detection. Its Hub repo is gated; the checkpoint was staged into the
  site's HF cache with `prefetch` from a login node.

Read the SMURF row with its coverage in mind: this is the `asr_long` run, and on
ST it translates `de/da/el` but echoes `cs/es/et/fi`, while Europarl `de->en`
comes back in German — a model behaviour the suite is meant to surface, not a
prompt mismatch (`FLEURS de->en` is a correct English translation).
