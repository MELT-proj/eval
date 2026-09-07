#!/bin/bash
#
# Build the melt-eval container image from infra/Singularity.def.
#
#   infra/setup/build_singularity.sh [output.sif]
#
# MUST be run from the repo root, with the training repo checked out as a
# sibling (../training): the def file's %setup rsyncs both `.` (this repo)
# and `../training` into the image, so the working directory is the build
# context. See README.md "Install" for why melt-eval needs the training repo
# at all (melt-proj, resolved via a relative path in pyproject.toml).
#
# Environment overrides:
#   SIF_OUT        output image path        (default: ./melt_eval_cuda126.sif; or $1)
#   BUILD_TMPDIR   scratch for the build    (default: $TMPDIR or /tmp)
#                  Needs ~2x the image size (~25GB free for the -devel base).
#   UV_CACHE_DIR   host uv cache to reuse   (default: none; bound read-write
#                  at /usr/local/share/uv/cache so wheels survive rebuilds)
#
# Requires apptainer (NOT singularity): the def file's %setup relies on
# APPTAINER_ROOTFS, which SingularityCE does not export. Sites that expose it
# via environment modules are handled below.
#
# The build pulls nvidia/cuda:*-devel from Docker Hub and downloads the whole
# python env (torch, transformers, etc. via melt-proj), so the build host
# needs outbound internet. Air-gapped clusters (e.g. MN5) should build
# elsewhere and copy the .sif over.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f infra/Singularity.def ]] || die "run this from the melt-eval repo root (infra/Singularity.def not found here)"
[[ -d ../training ]] || die "../training not found -- checkout the training repo as a sibling of melt-eval (see README.md 'Install')"

# Some sites ship apptainer as an environment module (e.g. Artemis).
if ! command -v apptainer >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/02-lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/02-lmod.sh
    fi
    if command -v module >/dev/null 2>&1; then
        module load apptainer 2>/dev/null || true
    fi
fi
command -v apptainer >/dev/null 2>&1 \
    || die "'apptainer' not found (try: source /etc/profile.d/02-lmod.sh && module load apptainer). singularity will NOT work: the def file needs APPTAINER_ROOTFS."
BIN=apptainer

SIF_OUT="${1:-${SIF_OUT:-./melt_eval_cuda126.sif}}"
BUILD_TMPDIR="${BUILD_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$BUILD_TMPDIR" "$(dirname "$SIF_OUT")"

# Unprivileged builds need a user namespace; skip the flag when already root.
FAKEROOT_ARGS=()
[[ "$(id -u)" -ne 0 ]] && FAKEROOT_ARGS=(--fakeroot)

# Optional uv cache bind: makes rebuilds much faster by reusing wheels.
BIND_ARGS=()
if [[ -n "${UV_CACHE_DIR:-}" ]]; then
    mkdir -p "$UV_CACHE_DIR"
    BIND_ARGS=(--bind "${UV_CACHE_DIR}:/usr/local/share/uv/cache")
fi

echo "[build] runtime: ${BIN} ($(${BIN} --version))"
echo "[build] output:  ${SIF_OUT}"
echo "[build] tmpdir:  ${BUILD_TMPDIR} ($(df -h "$BUILD_TMPDIR" | awk 'NR==2 {print $4}') free)"
echo "[build] uvcache: ${UV_CACHE_DIR:-<none>}"
echo "[build] training: $(cd ../training && pwd)"

export APPTAINER_TMPDIR="$BUILD_TMPDIR" TMPDIR="$BUILD_TMPDIR"

"$BIN" build "${FAKEROOT_ARGS[@]}" "${BIND_ARGS[@]}" --force "$SIF_OUT" infra/Singularity.def

echo "[build] done: ${SIF_OUT} ($(du -h "$SIF_OUT" | cut -f1))"
