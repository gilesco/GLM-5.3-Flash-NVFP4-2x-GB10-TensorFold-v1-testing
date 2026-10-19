#!/usr/bin/env bash
# shellcheck shell=bash disable=SC2034  # a config file: several vars are consumed by the scripts that source it
# Shared settings for start.sh, stop.sh, status.sh and scripts/*.sh. A setting's value comes from
# the first of these that sets it:
#   1. the environment: `PORT=9000 ./scripts/start.sh`
#   2. scripts/local.sh (this setup's own values; sourced as bash), then ./.env (KEY=value lines,
#      read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
#
# This recipe serves GLM-5.3-Flash (NVIDIA's NVFP4 checkpoint) on the native TensorFold CUDA
# engine ("tensorfold-native") at tensor parallel over two GB10 hosts, after PR #804 (first
# shipped in the v1.0.x line) with #855/#976's image and video input on top (the v1.1.0 line).
# Everything runs in Docker; no Python engine and no patch stack.
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# ---------------------------------------------------------------------------
# TensorFold version. Releases are frequent; "latest-v1" (the old default) resolves to the
# newest v1.* release at prepare time and records the resolved tag and commit in the prepared
# marker, so an upstream release changes nothing until you re-run prepare. Pin instead with
# e.g. TF_REF=v1.1.0 (a tag) or TF_COMMIT=<sha> (an exact tree; wins over TF_REF).
#
# Pinned to v1.1.0 for the ashhart/TensorFold#971 media-receipt window: v1.1.0's tag peels to
# 76aafcf274e60a5e31a59ce145b2f71f7b366e9d, the commit upstream's 2026-10-19 comment names
# (#855's images and #976's video). prepare records the resolved commit in the prepared marker;
# revert to latest-v1 (or bump the tag) once the window is done.
# ---------------------------------------------------------------------------
TF_REF="${TF_REF:-v1.1.0}"
TF_COMMIT="${TF_COMMIT:-}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"

# The GLM-5.3-Flash NVFP4 checkpoint (ModelOpt, read as stored; PR #804's family reads no other
# format). Pinned to the revision upstream's #971 thread agreed for the receipts.
# Alternatives:
#   09b04e5e... - the revision this site served first (also in both caches); it differs from
#                 da920bb0 only in config.json and hf_quant_config.json (layer 45 added to the
#                 ignore list), so the da920bb0 fetch reuses every 190 GiB blob and takes seconds
#   423acf37... - the revision PR #804's receipts were measured on (download it if you want the
#                 exact validated tree; see docs/VALIDATION.md)
# The ranks cross-check every checkpoint byte against each other at start, so a copy that
# differs on one host stops both before serving.
MODEL_ID="${MODEL_ID:-nvidia/GLM-5.3-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-da920bb0b9f4a06727223a349e55468e38352348}"

# The pair: this machine (the head) serves rank 0 and the API; WORKER — an ssh destination,
# key-based — runs rank 1. Rank 1 is started first, rank 0 serves HTTP (the native engine's
# rank-over-host pattern; docs/DELTAS.md). TP=3 is not servable on the native engine yet
# (it waits on upstream's multi-rank bootstrap fabric), so TP is fixed at 2. WORKER and the
# fabric addresses below are site settings: put them in scripts/local.sh
# (scripts/local.sh.example explains how to find each one).
TP=2
WORKER="${WORKER:-}"

# The point-to-point rails between the pair (this recipe was written on two ConnectX-7s, one
# /30 subnet per cable; any fabric with host-to-host reachability works). NCCL's bootstrap
# listens on the head's rail address and discovers the rest from there. Where every rank pair
# can reach every other, the default split exchange works and TF_GLM_PREFILL_REDUCE=gather is
# NOT needed (that is for fabrics where some rank pairs cannot connect).
HEAD_FABRIC_ADDR="${HEAD_FABRIC_ADDR:-}"                  # the head's rail address (required)
WORKER_FABRIC_ADDR="${WORKER_FABRIC_ADDR:-}"              # the worker's rail address (documentation)
NCCL_PORT="${NCCL_PORT:-29741}"
# Comma-separated netdevs NCCL may use. Empty: NCCL chooses — fine on single-interface hosts;
# on multi-homed ones, pin the rail netdevs in local.sh or NCCL may ride the LAN. NCCL_IB_HCA
# is empty on purpose (all RDMA devices); set it in local.sh to pin devices.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
# Extra NCCL debug, for a first bring-up: set NCCL_DEBUG=INFO in local.sh and watch the logs.

# Docker. The builder image compiles tensorfold-native (Zig + nvcc, sm_121); the runtime image
# carries the binary, its libraries and NCCL and is what both ranks run. The builder needs a
# CUDA devel base with nvcc for aarch64: nvcr.io/nvidia/pytorch:26.07-py3 is already pulled on
# the head and ships both, but any arm64 CUDA 13.x devel image works (set BASE_IMAGE).
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"                     # TensorFold's .zig-version (still 0.17.0 at v1.1.0)
SM_ARCH="${SM_ARCH:-121}"                                # GB10
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE_NAME="${IMAGE_NAME:-glm53-native-tf}"   # the runtime image; prepare.sh tags it <name>:<resolved version>
CONTAINER_NAME="${CONTAINER_NAME:-glm53-native-tf}"      # the same name on both Sparks

# Checkpoint and binary locations (host paths; the container mounts the HF cache read-write so
# the snapshot layout, blob names and hashes stay exactly as Hugging Face wrote them).
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
MODEL_SNAPSHOT="$HF_CACHE/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"

# Serving. The context the PR's receipts ran at; larger windows fit but are unmeasured on this
# pair (the engine's bf16 KV takes more per token than the old stack's fp8 latent cache) —
# docs/VALIDATION.md has the fit test. PARALLEL is the stream count (the lanes core; the PR
# served --parallel 2 and 4, and admitted 8).
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"             # the same port production serves on: clients keep one endpoint,
                                 # and the two stacks are mutually exclusive (start.sh's guard
                                 # refuses a busy port instead of fighting for it)
CONTEXT="${CONTEXT:-40960}"
PARALLEL="${PARALLEL:-4}"
DRAFTS="${DRAFTS:-1}"            # 0 passes --no-drafts (the serial reference; MTP is the only drafter here)
VISION="${VISION:-1}"            # --vision on both ranks (image_url/video_url parts in user
                                 # messages): the #971 media receipts need it, and the ranks must
                                 # agree on it before loading. 0 reproduces the text-only v1.0.12
                                 # behavior; a server without --vision refuses image parts.
VISION_URLS="${VISION_URLS:-0}"  # 1 adds --vision-urls: the server fetches public HTTPS image
                                 # URLs itself. data: URLs always work with --vision alone.
API_KEY="${API_KEY:-}"           # recommended; also required by upstream for --slide-style features
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-}"   # -> TENSORFOLD_CUDA_MEMORY_LIMIT_GB, empty: engine default

# Runtime state: pidfiles, logs (each rank's docker log saved gzipped at stop, LOG_KEEP kept),
# and the prepared marker that start.sh compares to decide whether prepare.sh must run again.
STATE_DIR="${STATE_DIR:-$HOME/.local/state/glm53-native}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/glm53-native/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
# The engine's file-digest cache, mounted into the container at /root/.cache/tensorfold: with it,
# a restart reads no shard (upstream #900: 0.1 s instead of the whole checkpoint); without it,
# every start re-reads ~190 GiB (~12 min on this pair's NVMe at ~270 MiB/s). One directory per
# host (the cache keys on each host's device/inode/times).
DIGEST_CACHE_DIR="${DIGEST_CACHE_DIR:-$HOME/.cache/glm53-native/tensorfold-digests}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

# The required site settings; scripts/local.sh (copied from scripts/local.sh.example) is where
# they belong. DRY_RUN is exempt: a dry run only prints the two docker commands and needs
# nothing. Everything else has a workable default.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -n "$WORKER" ]] || die "WORKER is not set - rank 1's ssh destination (key-based); fill it in scripts/local.sh (scripts/local.sh.example shows how to find it)"
  [[ -n "$HEAD_FABRIC_ADDR" ]] || die "HEAD_FABRIC_ADDR is not set - the head's rail address, where NCCL's bootstrap listens; fill it in scripts/local.sh (scripts/local.sh.example shows how)"
fi

worker() { ssh -o BatchMode=yes "$WORKER" "$@"; }  # args ARE the remote command (a bare ssh
                                                   # would eat our stdin as shell input)

# The docker run contract, shared by both ranks (the same as the EXL3 recipe's: host networking
# for the CX7 rails, the RDMA devices, locked memory for NCCL's registered buffers).
RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband
          --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864)

# Environment passed into both ranks: offline serving from the mounted cache, plus every
# TENSORFOLD_/TF_GLM_/NCCL_ variable this recipe sets.
env_args() {
  ENV_ARGS=(-e HF_HUB_OFFLINE=1)
  [[ -z "$NCCL_SOCKET_IFNAME" ]] || ENV_ARGS+=(-e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME")
  [[ -z "$NCCL_IB_HCA" ]] || ENV_ARGS+=(-e NCCL_IB_HCA="$NCCL_IB_HCA")
  [[ -z "$MEMORY_LIMIT_GB" ]] || ENV_ARGS+=(-e TENSORFOLD_CUDA_MEMORY_LIMIT_GB="$MEMORY_LIMIT_GB")
  while IFS='=' read -r name _; do ENV_ARGS+=(-e "$name=${!name}"); done \
    < <(env | grep -E '^(TENSORFOLD|TF_GLM)_[A-Z0-9_]+=' || true)
}

# The serve arguments both ranks take (rank 0 adds --name/--host/--port in start.sh). The ranks
# must agree on window, streams and the vision flags (all of them) before either loads: each
# allocates its own state before rank 0 admits, and the media limits are part of the handshake.
serve_args() {
  local args=(serve "/root/.cache/huggingface/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
              --context "$CONTEXT" --parallel "$PARALLEL")
  [[ "$DRAFTS" == 1 ]] || args+=(--no-drafts)
  if [[ "$VISION" == 1 ]]; then
    args+=(--vision)
    [[ "$VISION_URLS" == 1 ]] && args+=(--vision-urls)
  fi
  echo "${args[@]}"
}

# The prepared marker: a one-line description of everything a start depends on. start.sh re-runs
# prepare.sh whenever the config fingerprint no longer matches (a new upstream release, another
# revision, another worker, a different window or stream count).
config_line() {   # the fingerprint: everything except the resolved version
  echo "model=$MODEL_ID@$MODEL_REVISION worker=$WORKER context=$CONTEXT parallel=$PARALLEL vision=$VISION head=$HEAD_FABRIC_ADDR"
}
prepared_line() { # the fingerprint plus what prepare resolved (written to $STATE_DIR/prepared)
  echo "tf=$TF_RESOLVED_REF@$TF_RESOLVED_COMMIT image=$IMAGE_REF $(config_line)"
}
