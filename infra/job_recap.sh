#!/bin/bash
#
# Recap of the last N SLURM jobs on a site, with QoS and resources used.
#
#   infra/job_recap.sh <artemis|mn5> [N]
#
# N defaults to 20. Reads sacct directly on artemis; over SSH (the same
# REMOTE_SSH alias sites/mn5.sh uses) on mn5.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <artemis|mn5> [N]}"
N="${2:-20}"

# -X: one row per job (skip .batch/.extern steps). AllocTRES carries
# cpu/mem/gpu counts and gpu type together, which is the single field worth
# having if only one is going to fit in a terminal.
SACCT_CMD='sacct -X -u "$USER" -S now-30days \
    --format=JobID%14,JobName%22,Partition%10,QOS%12,State%12,Elapsed%12,AllocTRES%60,Start%20'

# Keep the two header lines, then the N most recent data rows (newest first).
FILTER='{ if (NR<=2) print; else data[NR]=$0 }
    END { c=0; for (i=NR; i>2 && c<N; i--) if (i in data) { print data[i]; c++ } }'

case "$SITE" in
    artemis)
        eval "$SACCT_CMD" | awk -v N="$N" "$FILTER"
        ;;
    mn5)
        ssh mn5 "$SACCT_CMD" | awk -v N="$N" "$FILTER"
        ;;
    *)
        die "unknown site '$SITE' (expected 'artemis' or 'mn5')"
        ;;
esac
