"""Prefetch the gated RedVox metadata and audio into the shared HF cache.

Run once on a login node with access to the gated dataset:

    export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
    export HF_HOME=/path/to/shared/hf-cache
    python scripts/prefetch_redvox.py

Compute jobs then read the cache; set ``HF_HUB_OFFLINE=1`` there if the nodes
have no outbound network. The dataset is gated, so the token is required for
this prefetch. Do not commit the token or any downloaded audio.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download


REPO = "FBK-MT/RedVox"
REVISION = "fca218d440f6b8ed4508bea2eee6b4fb5a913e22"
PATTERNS = ["*/metadata.jsonl", "*/audio/*.wav"]


def main() -> str:
    """Download the pinned RedVox revision and return the cache snapshot path."""
    path = snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=PATTERNS,
    )
    print(path)
    return path


if __name__ == "__main__":
    main()
