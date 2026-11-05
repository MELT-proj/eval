#!/bin/bash
#
# Build the post-hoc metrics venv: MCIF's official scorers (WER/COMET/BERTScore)
# and COMET, for `evaluate.py score`.
#
#   infra/setup/metrics_venv.sh [VENV_DIR]
#
# VENV_DIR defaults to the directory of $METRICS_VENV_PATH (the activate script
# the site files point at), e.g. $PROJECT/$USER/venvs/metrics-eval. Run from the
# repo root; it installs this checkout (`-e .`) into the venv.
#
# `mcif-bench` is installed from git, not PyPI: the 1.x releases drive RWTH's
# mwerSegmenter binary, which is x86-only, while the 2.0 line resegments with
# `mweralign` in-process -- no external tool, no MWERSEGMENTER_ROOT, and it
# builds from source on aarch64 (gcc/g++, nothing else).
#
# Deliberately no --prerelease=allow: that resolves httpx to its 1.0.dev* line,
# which dropped httpcore, and every scoring job then dies on `import inspect_ai`
# with ModuleNotFoundError: No module named 'httpcore'.
#
# Environment overrides:
#   MCIF_REPO        mcif-bench source        (default: the upstream git URL)
#   MCIF_COMMIT      revision to install      (default: the 2.0-line commit)
#   METRICS_PYTHON   interpreter for the venv (default: 3.12)
#   UV_PYTHON_INSTALL_DIR  where uv keeps managed interpreters (default: next
#                   to the venv, never $HOME -- compute nodes may not mount it)
#
# Needs internet (git + PyPI). The COMET/BERTScore checkpoints and the MCIF
# reference files are separate: cache those from a login node too, see
# docs/jupiter-continuous-eval.md "Offline models for the post-hoc scorers".

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

MCIF_REPO="${MCIF_REPO:-https://github.com/hlt-mt/mcif.git}"
MCIF_COMMIT="${MCIF_COMMIT:-cdb1d62a00281a14bef52f829b113fc73981c85d}"
METRICS_PYTHON="${METRICS_PYTHON:-3.12}"

[[ -f pyproject.toml && -d melteval ]] || die "run this from the melt-eval repo root (pyproject.toml + melteval/ not found here)"
command -v uv >/dev/null 2>&1 || die "uv not found in PATH (see README.md 'Install')"

VENV_DIR="${1:-}"
if [[ -z "$VENV_DIR" && -n "${METRICS_VENV_PATH:-}" ]]; then
    VENV_DIR="$(dirname "$(dirname "$METRICS_VENV_PATH")")"
fi
[[ -n "$VENV_DIR" ]] || die "usage: $0 <venv dir>  (or export METRICS_VENV_PATH=<venv>/bin/activate)"
if [[ "$VENV_DIR" == "$HOME"/* && -z "${ALLOW_VENV_IN_HOME:-}" ]]; then
    die "refusing to build a venv under \$HOME ($VENV_DIR): compute nodes may not mount it; use scratch/project storage (ALLOW_VENV_IN_HOME=1 overrides)"
fi

# Keep the interpreter next to the venv (not in ~/.local/share/uv), for the same
# reason: a $HOME python is invisible to a compute node that does not mount $HOME.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "$VENV_DIR")/uv-python}"

echo "[metrics_venv] venv:   $VENV_DIR"
echo "[metrics_venv] mcif:   $MCIF_REPO@$MCIF_COMMIT"
echo "[metrics_venv] python: $METRICS_PYTHON"

# --clear: a half-installed metrics venv is what made scoring fail silently on
# JUPITER (an httpx pre-release without httpcore); rebuild rather than repair.
uv venv --clear --python "$METRICS_PYTHON" --python-preference only-managed "$VENV_DIR"

VIRTUAL_ENV="$VENV_DIR" uv pip install -e . \
    "mcif-bench @ git+${MCIF_REPO}@${MCIF_COMMIT}" \
    "setuptools<81" "transformers<5"

# What the scoring jobs actually import, plus the two failure modes that do not
# show up until a job is already on a GPU: a missing httpcore (inspect_ai) and
# pkg_resources (comet's torchmetrics pin).
"$VENV_DIR/bin/python" - <<'EOF'
import httpx, httpcore  # noqa: F401  -- inspect_ai imports httpcore unconditionally
from mcif.evaluation import score_asr, score_st, score_sqa  # noqa: F401
import mweralign, comet, melteval  # noqa: F401
print("ok", "httpx", httpx.__version__)
EOF
