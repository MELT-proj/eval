# Frozen sets

A frozen set is the answer to "which samples is this number computed over?".
It is a directory with two files:

```
manifest.jsonl    one JSON record per sample
frozen_set.json   spec, spec hash, counts, hours, drop reasons, reader versions
```

## Why audio is referenced, not copied

The obvious design is to export the chosen samples as wav files: portable,
self-contained, reproducible. It is the wrong default here, on disk grounds.

Measured on artemis in August 2026: scratch was at 96 % (2.3 T free of 56 T) and
`/mnt/home` had 28 G. Individual Shar test splits run 100 M–1.3 G *compressed*
each, and decoding to 16 kHz 16-bit wav inflates that. A multi-language ASR+ST
suite would cost tens of GB to hold a second copy of audio that already exists
on the same filesystem.

So a record carries an `audio` **locator** instead:

```json
{"kind": "shar", "dir": "/…/fleurs/de_de/test", "index": 412}
```

A frozen set of 80 samples is 37 KB. Ten thousand samples cost about 5 MB.

Resolution happens one batch at a time inside the model provider, through
lhotse's indexed reader, which gives O(1) random access by global cut index.
Warm cost is roughly 2 ms for the metadata seek and 6 ms to decode, against
~18 ms if access is random rather than sequential — which is why records are
stored sorted by index, so a run reads each shard once, in order.

### The invariant this rests on

The index in a locator is the cut's position in **manifest iteration order**,
and it is resolved through **indexed random access**. Those are two different
code paths through lhotse. They agree today, and
`tests/test_shar_integration.py::TestLocatorResolution` pins that down by
checking the resolved cut's ID against the one recorded at freeze time.

If they ever diverged, every sample would be scored against another sample's
audio, and nothing would raise — the run would simply report a bad number. That
is why the check compares IDs rather than just asserting the audio loads.

### What you give up

A manifest is only valid while its sources are intact. There is no checksum over
the audio, so a corpus rebuilt in place will not be detected. Mitigations:
`frozen_set.json` records the source paths, a spec hash, and per-sample duration
and reference text, so drift shows up as a duration mismatch rather than
silently.

Frozen sets are also not portable across machines, because locators hold
absolute paths.

## When to materialise instead

`--materialize-audio` (planned) writes 16 kHz mono wav and rewrites locators to
point at the files. Use it when portability is the point:

- handing an identical eval set to Smurf on another cluster;
- staging a set to air-gapped MN5;
- publishing a benchmark subset alongside results.

## Sample identity

`sample_key` is `<source_index>-<ordinal>`, assigned by the freezer. It is
deliberately **not** the corpus's own ID: FLEURS cut IDs are small integers that
repeat within a language and across languages
([MELT-proj/training#54](https://github.com/MELT-proj/training/issues/54)), so
keying an eval log by cut ID would merge distinct samples without complaint. The
original ID is kept in `cut_id` for provenance.

## Budgeting

Everything is reported in **audio hours**, from `cut.duration`, which every cut
carries. `custom.num_tokens` is absent from most sources
([#59](https://github.com/MELT-proj/training/issues/59)), so any token-based
budget silently reads as zero.
