# UC3M TSC "mollete" cluster (login node kumo.tsc.uc3m.es; compute nodes
# kumo02 = 1x RTX 4090 24 GB, kumo03/kumo04 = 2x RTX 5090 32 GB each).
#
# The venv, the HF cache and every output therefore live under
# /export/usuarios_ml4ds/$USER, and so does inspect's own state (XDG_DATA_HOME),
# which would otherwise land in a $HOME the job's node may not see.
#
# The RTX 5090s (sm_120, Blackwell) need a torch built against CUDA >= 12.8;
# a cu126 wheel imports fine and then fails on the first kernel there. Build
# the venv with a cu128 torch. Jobs ask for a 5090 by default (MELT_GPU_TYPE
# below); MELT_GPU_TYPE=RTX4090 sends one to kumo02 instead.
#
# 24-32 GB per GPU: 7B-class models fit in bf16, Qwen3-Omni (~70 GB) does not.

_mollete_root="/export/usuarios_ml4ds/${USER}"

# --- python environment ------------------------------------------------------
# A SMURF venv (NeMo + speechllm + melt-eval[metrics]) on the shared disk.
# Does not exist until someone builds it -- see README.md "Install".
export VENV_PATH="${VENV_PATH:-${_mollete_root}/venvs/smurf-eval/bin/activate}"
# Post-hoc metrics (MCIF's official scorers, COMET): mcif-bench + unbabel-comet +
# melt-eval, kept apart because they pin their own transformers.
export METRICS_VENV_PATH="${METRICS_VENV_PATH:-${_mollete_root}/venvs/metrics-eval/bin/activate}"
# mwerSegmenter, needed by MCIF's official translation scorer (a 32-bit x86
# binary: runs here, not on an aarch64 machine).
export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${_mollete_root}/tools/mwerSegmenter}"

# --- storage (host paths) ------------------------------------------------------
# Keep the login shell on the same HF cache (`export HF_HOME=...` in ~/.bashrc),
# or a download made on kumo is not the one the job reads.
export HF_HOME="${HF_HOME:-${_mollete_root}/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-${_mollete_root}/eval-logs}"
# Where scripts/evaluate.py puts each campaign (configs/eval/*.yaml: output_dir).
export EVAL_ROOT="${EVAL_ROOT:-${_mollete_root}/eval-runs}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${_mollete_root}/.local/share}"
export TMPDIR="${TMPDIR:-${_mollete_root}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"

# --- misc ----------------------------------------------------------------------
# Compute nodes are not forced offline; set HF_HUB_OFFLINE=1 yourself once the
# cache is warm and you want a run pinned to it.

# --- scheduler -------------------------------------------------------------
# Account ml4ds. QoS tiers (`sacctmgr show qos`):
#   debug   30 min, 1 GPU, at most 2 submitted jobs per user
#   normal  2 days, 2 GPUs per user   <- default
#   long    7 days, lowest priority, preemptible
# Both partitions cap a job at 10 CPUs per node, and the default memory is
# 4 GB per CPU -- not enough to load a 7B checkpoint on the host side, hence
# the explicit --mem.
#   MELT_QOS=debug MELT_TIME=00:20:00 infra/runners/submit_eval.sh mollete …
#
# The `normal` 2-GPUs-per-user cap means a third job just waits as PENDING
# (QOSMaxGRESPerUser), which is harmless; debug's 2-job submit cap is not --
# sbatch rejects the third -- so scripts/evaluate.py is told to wait for room.
export MELT_GPU_TYPE="${MELT_GPU_TYPE-RTX5090}"   # set it empty for any GPU

if [[ "${MELT_QOS:-normal}" == debug ]]; then
    export MAX_QUEUED="${MAX_QUEUED:-2}"
fi

SBATCH_ARGS=(
    --account=ml4ds
    --partition="${MELT_PARTITION:-gpu}"
    --qos="${MELT_QOS:-normal}"
    --time="${MELT_TIME:-04:00:00}"
    --nodes=1
    --ntasks=1
    --gres="gpu:${MELT_GPU_TYPE:+${MELT_GPU_TYPE}:}1"
    --cpus-per-task="${MELT_CPUS:-8}"
    # 128G: a SMURF checkpoint is loaded on the host (map_location=cpu) -- the
    # fp32 EuroLLM-9B (~36 GB) plus the 21 GB checkpoint -- before the GPU sees it.
    --mem="${MELT_MEM:-128G}"
)
