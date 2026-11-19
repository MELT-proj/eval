# PolySpeech-100

[PolySpeech-100-v1](https://huggingface.co/datasets/youngseng/PolySpeech-100-v1)
(CC-BY-NC-4.0, public) is a **spoken multiple-choice reading-comprehension**
benchmark in the style of Belebele: a passage, a question and four options per
sample, 110 languages. Passage, question and each option exist both as audio and
as text; the reference is one option number. Scoring is accuracy (chance 25%),
so no judge model is involved.

We evaluate 24 of the languages -- the official EU languages the dataset has,
plus Ukrainian and Russian (Croatian and Irish are not in it) -- in two variants:

| Variant | Audio the model hears | In the prompt as text |
|---|---|---|
| `pa_qa` | passage, a 1 s pause, then the spoken question | the options (and a fixed scaffold sentence in place of the question) |
| `pa_qt` | passage | the question and the options |

The options are **always text**. In `pa_qa` the question text never reaches the
model (a test pins this); the real one is kept in the sample's metadata
(`question_text`) for analysis only. Both variants are the `audio_mcq` task
scored by `mcq_scorer`, which reports `accuracy_<dataset_id>` per language and
`unresolved_rate` (completions no option could be read out of).

## Running

```bash
export POLYSPEECH_ROOT=/e/data1/smurf4eu-data/speech/polyspeech-100    # on jupiter
inspect eval melteval/tasks.py@audio_mcq --model melt/<checkpoint> \
  -T spec=configs/hf/polyspeech-100-pa_qa.yaml -T dataset_id=polyspeech-pa_qa-de
```

`configs/hf/polyspeech-100-{pa_qa,pa_qt}.yaml` have one source per language, so
`-T dataset_id=polyspeech-<variant>-<lang>` (or `-T lang=<lang>`) reads only that
language. Without it all 24 are pooled (19,224 samples) -- the per-language
accuracies are still reported, but one headline number across languages hides
more than it shows. The spec is read live; like AIR-Bench it pins an upstream
revision, which the reader checks against the reconstruction's `README.md`.
Prompts follow the AIR-Bench `audio_mcq` layout (letters, "Answer with the letter
of the correct option."); `instruction_template` on a source overrides it.

## Dev subset

`configs/hf/polyspeech-100-pa_qt-200.yaml` is the `pa_qt` spec with
`max_samples: 200` and `seed: 0` on every source: 4,800 samples, ~107 h. It is
the dev SQA/SLU row, frozen as `POLYSPEECH_PA_QT_SET` and consumed by the
separate suite `configs/eval/suites/speechval-polyspeech.yaml` (not
`speechval-dev.yaml`, which artemis runs). It is a screening subset, **not
comparable with full-set PolySpeech numbers**; the 200 are a seeded draw after
the pa_qt validity filter, so `pa_qa` would pick a slightly different 200.
`scripts/frozen_length_stats.py` reports the selected subset's duration
distribution — the passages are long-form (89% of the full set exceeds SMURF's
40 s training window), so read accuracy with `unresolved_rate`.

## What `mcq_scorer` accepts

The reference is the option's **text**, so the scorer has to read an option out
of a free-form completion. `resolve_choice` tries these, most explicit first:

| form | example |
|---|---|
| a leading label | `B`, `B.`, `(b)`, `b:` |
| a capitalised label the completion ends on | `The answer is B.` |
| an explicit answer/option/choice statement | `The correct answer is **C. …**`, `The correct answer is: D`, `The correct option is B`, `The correct choice is C` |
| the option text alone | `Nonbinary` |
| exactly one option's text inside a sentence | `I think the speaker is nonbinary.` |

Text matching ignores case and punctuation. Three guards matter in practice:
the letter in the trailing and explicit forms must be **capitalised** (`the
answer is a bird` names the article, not option A); two different explicit
letters (`the answer is A … the answer is B`) are a contradiction and resolve
to nothing; and a completion that mentions several options without an explicit
letter is unresolved. An explicit letter outranks option text when they
disagree. `unresolved_rate` reports the share of completions none of this could
read.

## The reconstruction the reader needs

Upstream ships 110 parquet files that are *file archives* (one row = one wav or
txt), not a table, so the reader does not use `datasets`. It reads a local
reconstruction made once by extracting them (on jupiter:
`/e/data1/smurf4eu-data/speech/polyspeech-100/`, whose `README.md` has the
commands, checksums and validation): a `manifest.jsonl` with one record per
sample and the extracted audio under `audio/`. Audio is referenced, not copied.

Facts about it that matter for results (all checked there, 24 languages x 801
samples, 567 h):

* **Which questions differ by language.** 17 languages (bg cs da el en fi fr it lv
  mt pl pt sk sl es sv uk) share the same 801 questions with identical answer
  keys. The other seven (de et hu lt nl ro ru) use a different 801 of Belebele's
  900; only about 370-425 questions overlap English. So this is not a parallel
  test set across all 24: compare languages on the overlap, joining on
  `metadata["index"]`/`["qid"]`.
* **The key is the directory name.** Upstream's `info.txt` has `Index`/`QID` lines
  that are wrong in those seven languages (answer agreement with English: ~25%
  by that key, 100% by the directory's). The reconstruction and this reader use the
  directory-derived `index`/`qid`.
* **Audio is heterogeneous.** Mono, but 16/24/44.1/48 kHz mixed within a language,
  and some MP3 inside `.wav` (all of mt and uk, parts of et pt ro ru). Each segment
  is decoded and resampled to 16 kHz on its own before joining.
* **Passages are long**: median 77 s, p95 143 s, max 260 s (per sample, summed
  segments), question 1.5-29 s. Models with a 30-40 s window (Qwen2-Audio, Gemma 4,
  Phi-4, DeSTA2.5) will have almost every sample marked "not evaluated" -- by design,
  rather than scored on a truncated passage -- so they cannot run this benchmark;
  size batches and wall time for the others accordingly.
* Mt and uk are entirely 24 kHz MP3 and much shorter (10-12 h per language, against
  19-31 h for the rest), and their text length tracks audio duration almost
  perfectly (r = 0.97+, against 0.65-0.83 elsewhere), which suggests a different
  generation process. Treat cross-language comparisons involving them with care.
* Not verified: that each audio file says what its text says (no ASR was run).
