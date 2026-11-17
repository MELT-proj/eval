"""The fondue_checkpoint.sh scheduler: argument checks, config editing, the plan.

The script only shells out to SLURM and ssh in its non-dry modes, so these
tests cover the parts that decide what would be run: ``--config-only`` edits a
config the way the real run would, ``--dry-run`` prints the phases, and bad
arguments stop before anything is scheduled.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fondue_checkpoint.sh"

CONFIG = """\
# A campaign config, cut down: the model anchor and the pack block are what the
# scheduler reads.
name: speechval-fondue
output_dir: /tmp/none

models:
  - name: fondue-step-17000
    model: melt//mnt/scratch-nyx/giuseppe/fondue-keep/checkpoint-17000
    venv: /venvs/melt-main/activate
    args: &fondue_args
      - -M
      - batch_size=8
  - name: fondue-step-26000
    model: melt//mnt/scratch-nyx/giuseppe/fondue-keep/checkpoint-26000
    venv: /venvs/melt-main/activate
    args: *fondue_args

benchmarks:
  - name: voxpopuli-dev
  - name: fleurs24-st-xen-dev

pack:
  time: "04:00:00"
  max_slices: 200
  gpus: 1
"""


#: What ``yaml.safe_dump`` writes as the campaign snapshot: sequence items at
#: column 0, and anchors named by the dumper rather than by hand. The scheduler
#: has to edit this file too, so it may not assume the hand-written style above.
GENERATED = """\
name: speechval-fondue
models:
- name: fondue-step-32000
  model: melt//mnt/scratch-nyx/giuseppe/fondue-keep/checkpoint-32000
  venv: /venvs/melt-main/activate
  args: &id001
  - -M
  - batch_size=8
benchmarks:
- name: voxpopuli-dev
pack:
  time: '02:00:00'
  max_slices: 200
  gpus: 1
"""


@pytest.fixture
def campaign(tmp_path):
    """A temp campaign config and a checkpoint root holding step 33000."""
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    root = tmp_path / "fondue-keep"
    (root / "checkpoint-33000").mkdir(parents=True)
    return {"config": config, "root": root, "before": config.read_text()}


def run_script(campaign, *args, check=True):
    env = {
        "PATH": "/usr/bin:/bin",
        "MELT_FONDUE_CONFIG": str(campaign["config"]),
        "MELT_FONDUE_CKPT_ROOT": str(campaign["root"]),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, check=check
    )


def test_config_only_adds_the_checkpoint_once(campaign):
    first = run_script(campaign, "33000", "--config-only")
    text = campaign["config"].read_text()
    assert text.count("- name: fondue-step-33000") == 1
    assert "model: melt//" in text and "checkpoint-33000" in text
    assert "args: *fondue_args" in text
    assert "- name: fondue-step-17000" in text and "- name: fondue-step-26000" in text
    assert "# A campaign config, cut down" in text  # comments survive the edit
    assert "33000" in first.stdout

    run_script(campaign, "33000", "--config-only")
    assert campaign["config"].read_text().count("- name: fondue-step-33000") == 1


def test_config_only_keeps_the_model_list_ordered(campaign):
    run_script(campaign, "33000", "--config-only")
    text = campaign["config"].read_text()
    models = [line.split("fondue-step-")[1] for line in text.splitlines() if line.strip().startswith("- name: fondue-step-")]
    assert models == ["17000", "26000", "33000"]
    assert text.index("benchmarks:") > text.index("fondue-step-33000")


def test_dry_run_prints_the_phases_and_changes_nothing(campaign):
    result = run_script(campaign, "33000", "--dry-run")
    out = result.stdout
    assert "librispeech-validation-clean" in out and "voxpopuli-dev" in out
    assert "fleurs24-st-xen-dev" in out
    assert "covost2-dev" in out and "europarl-st-dev" in out
    assert "evaluate.py score" in out and "metricx" in out.lower()
    assert campaign["config"].read_text() == campaign["before"]


def test_no_score_skips_the_metricx_step(campaign):
    result = run_script(campaign, "33000", "--dry-run", "--no-score")
    assert "europarl-st-dev" in result.stdout
    assert "metricx" not in result.stdout.lower()
    assert "evaluate.py score" not in result.stdout


def test_rejects_a_step_that_is_not_a_number(campaign, tmp_path):
    result = run_script(campaign, "much-later", "--dry-run", check=False)
    assert result.returncode != 0
    assert "step" in (result.stderr + result.stdout).lower()


def test_rejects_a_checkpoint_that_is_not_there(campaign):
    result = run_script(campaign, "99000", "--dry-run", check=False)
    assert result.returncode != 0
    assert "99000" in (result.stderr + result.stdout)
    assert campaign["config"].read_text() == campaign["before"]


def test_requires_a_step_argument(campaign):
    result = run_script(campaign, "--dry-run", check=False)
    assert result.returncode != 0
    assert "usage" in (result.stderr + result.stdout).lower()


def test_config_only_handles_a_generated_config(tmp_path):
    """The campaign config is safe_dump output: column-0 items, dumper anchors."""
    import yaml

    config = tmp_path / "config.yaml"
    config.write_text(GENERATED)
    root = tmp_path / "fondue-keep"
    (root / "checkpoint-45000").mkdir(parents=True)
    env = {
        "PATH": "/usr/bin:/bin",
        "MELT_FONDUE_CONFIG": str(config),
        "MELT_FONDUE_CKPT_ROOT": str(root),
    }
    subprocess.run(
        ["bash", str(SCRIPT), "45000", "--config-only"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    doc = yaml.safe_load(config.read_text())  # the edit must leave valid YAML
    assert [m["name"] for m in doc["models"]] == ["fondue-step-32000", "fondue-step-45000"]
    added = doc["models"][1]
    assert added["model"] == "melt/" + str(root / "checkpoint-45000")
    assert added["venv"] == "/venvs/melt-main/activate"
    assert added["args"] == ["-M", "batch_size=8"]
    assert [b["name"] for b in doc["benchmarks"]] == ["voxpopuli-dev"]
    assert doc["pack"]["time"] == "03:00:00"


def test_config_only_repairs_a_broken_entry(tmp_path):
    """An entry written by the old, style-blind version is replaced, not kept."""
    import yaml

    broken = GENERATED.replace(
        "benchmarks:",
        "  - name: fondue-step-45000\n"
        "    model: melt//mnt/scratch-nyx/giuseppe/fondue-keep/checkpoint-45000\n"
        "    venv: /venvs/melt-main/activate\n"
        "    args: *fondue_args\n"
        "benchmarks:",
    )
    config = tmp_path / "config.yaml"
    config.write_text(broken)
    root = tmp_path / "fondue-keep"
    (root / "checkpoint-45000").mkdir(parents=True)
    env = {
        "PATH": "/usr/bin:/bin",
        "MELT_FONDUE_CONFIG": str(config),
        "MELT_FONDUE_CKPT_ROOT": str(root),
    }
    subprocess.run(
        ["bash", str(SCRIPT), "45000", "--config-only"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    doc = yaml.safe_load(config.read_text())
    assert [m["name"] for m in doc["models"]].count("fondue-step-45000") == 1
    assert doc["models"][-1]["args"] == ["-M", "batch_size=8"]
