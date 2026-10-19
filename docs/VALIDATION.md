# Validation gate (run when brain02/brain03 are free)

Nothing here is trusted until these pass on our own pair. The commands assume the server from
`scripts/start.sh` is up and a TensorFold checkout of the same version is available for its
python bench tools (they are standard-library clients; `tools/bench_openai.py`,
`tools/bench_concurrent.py`, `tools/prefill_cold.py` in the TensorFold tree).

## 0. Before the GPU work

- `scripts/prepare.sh` clean from an empty state; `docker run --rm <image> --version` OK.
- `DRY_RUN=1 scripts/start.sh` prints the two docker commands as expected.

## 1. Bring-up

- [ ] `scripts/start.sh` reaches `/health`; both containers stay up 10 minutes idle.
- [ ] Rank logs show the checkpoint digest exchange (both ranks agreeing on bytes) and both
      ranks' post-load memory (`/memory` if the route exists; `docker stats` otherwise).
- [ ] `scripts/stop.sh` stops both and leaves two gzipped logs in `$LOG_DIR`.

## 2. Exactness (the same gates the PR used, through the API)

- [ ] **Drafted = plain**: one prompt at temperature 0 with drafts on and with `DRAFTS=0`
      (or `--no-drafts`) — same tokens.
- [ ] **Resumed = fresh**: a two-turn conversation vs the same second turn sent to a server
      started without the history — same tokens (the prompt cache's snap hooks).
- [ ] **Concurrent = solo**: 4 concurrent requests vs each run alone — same tokens
      (`bench_concurrent.py --alone --serial` against the server).
- [ ] Sampled request at a seed twice — deterministic across the two ranks.

## 3. Capacity

- [ ] `/memory` (or `/health`) peak device memory at `--context 40960 --parallel 4` ≤ ~100 GiB
      on **both** ranks (the PR saw 97.3 GiB on rank 0 at `--parallel 2`).
- [ ] Long-context fit probe: raise `CONTEXT` stepwise (131072, 262144, 524288) on a restart;
      record where admission refuses, and the free memory left. The old stack served 1M tokens
      on fp8 KV; bf16 KV will refuse earlier — find the knee before promising anything.
- [ ] One rank under `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=21` less than the other refuses streams
      naming that rank (the PR's admission check).

## 4. Speed (same method as the old repo's README table, for a fair row)

- [ ] sparkDash or `bench_concurrent.py`: aggregate + per-request tok/s and TTFT at 1/2/4
      concurrent, prose and code prompts, GPU clocks capped at 2,200 MHz as before.
- [ ] `prefill_cold.py`-style cold prompt reading at 2k/8k/32k/65k/131k.
- [ ] Compare against the old table in the EXL3 recipe's README; record in this file.

| Metric | EXL3 production (measured 2026-10-11) | NVFP4 native (fill in) |
| --- | --- | --- |
| decode 1 / 2 / 4 concurrent (prose) | 61.8 / 75.2 / 105.5 tok/s | |
| prefill 8k / 32k / 131k | 2055 / 2018 / 1900 tok/s | |
| TTFT 8k | 4.20 s | |

## 5. Client compatibility

- [ ] hermes agent: custom endpoint `http://<head>:8000/v1`, model `GLM-5.3-Flash-NVFP4`;
      a tool-calling loop works end to end (the native server has its own tool-call parsing —
      check streamed calls arrive whole, as the old stack's GlmCallStreamer did).
- [ ] `toolcheck.py` / `tool-eval-bench --perf` from the old repo's tools against this server.
- [ ] Structured outputs (`response_format`), `/tokenize`, `/v1/models`.

## 6. Engine-level checks (optional, in the builder image)

The runtime image ships no test programs (upstream's packaging omits them). To run the PR's
`tf-cuda-test` checks (`glm-ref`, `glm-shared-check`, `glm-lanes-concurrent`, `glm-lanes-cover`),
open a shell in the build stage of the same commit on one Spark:

```sh
docker run --rm -it --gpus all <builder-stage-image> bash   # see image/build-native.sh for the layout
zig build test-cuda-host && zig build test                   # host tests need no GPU
tf-cuda-test glm-ref && tf-cuda-test glm-shared-check        # on the GPU, TP=1 on one host
```

The TP=2 lane checks ran at build time on the PR author's pair; re-running them here is the
last word on our hardware.

## 7. Media receipts for [ashhart/TensorFold#971](https://github.com/ashhart/TensorFold/issues/971)

What upstream's 2026-10-19 comment owes from this pair: the NVFP4 checkpoint's media checks on
two GB10 ranks — **drafted/plain, changed-image cache isolation, resumed/fresh, concurrent/solo**,
plus video if the window permits — at the 40,960-token/four-stream window, **recording startup
admission and per-rank peak memory**. The Python-engine comparison is optional (that line is
frozen); the EXL3 run and SvangenStudios' NVFP4 text-only report do not cover these.

The server needs `--vision` (both ranks — `VISION=1`, the default since the pin). Media parts are
OpenAI-style `image_url` / `video_url` content parts with `data:` base64 URLs (HTTPS URLs need
`VISION_URLS=1`). `scripts/media_gates.py` runs the gates end to end and prints a receipt block:

```sh
python3 scripts/media_gates.py --base-url http://127.0.0.1:8000 --model GLM-5.3-Flash-NVFP4
# --video /path/to/clip.mp4   add the video gates (any h264/hevc/vp9/mp4 clip; see below)
```

- [ ] **Startup admission + per-rank memory (before any media work).** Save both ranks' startup
      log excerpt — the admission block where each rank reports its memory after load, and the
      agreed `--vision` handshake — plus `/health`'s memory block (`python3 -m json.tool`), and
      `docker stats --no-stream` for the record (meaningless for device allocations on GB10's
      unified memory, but part of the paper trail). The bf16 tower lands unquantized on rank 0;
      the deltas against the v1.0.12 numbers (model 93.72 GiB/rank, 0.89 GiB/stream, 100.6 GB
      peak, ~21 GiB spare) are the interesting line in the receipt.
- [ ] **Drafted = plain with images.** The same image request at temperature 0 twice — once with
      drafts on (default) and once with `"draft": false` — identical content.
- [ ] **Changed-image cache isolation.** Two conversations identical except the image: turn 1
      (image + "what colour is the square?"), turn 2 (follow-up). Each conversation's turn 2 must
      answer about *its own* image — a stale prefix served past the swapped image fails this.
      `media_gates.py` uses a red and a blue square and checks the answers differ per colour.
- [ ] **Resumed = fresh past an image.** A continued conversation's turn-2 reply (prompt cache
      resumes past the image rows) vs the same turn-2 sent as one fresh request containing the
      full history — identical content.
- [ ] **Concurrent = solo with images.** Four image requests in flight together, each equal to
      its own reply run alone (`media_gates.py` does this with threads; upstream's
      `bench_concurrent.py --alone --serial` style).
- [ ] **Video (if the window permits).** Any small clip whose motion is easy to describe (the
      direction something moves, a colour that changes). The builder base has no ffmpeg; make a
      clip in a throwaway container, or bring a file:
      `docker run --rm -v $PWD:/w ubuntu:24.04 bash -c "apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null && ffmpeg -loglevel error -f lavfi -i testsrc=duration=2:size=256x256:rate=8 -pix_fmt yuv420p /w/clip.mp4"`.
      Gates: a video reply that describes the clip correctly, drafted = plain on the video
      request, concurrent videos = solo.
- [ ] Post the receipt block to #971: gate results, both ranks' memory lines, the resolved
      `tf=` marker line from `cat $STATE_DIR/prepared`, and the checkpoint revision.

Note: the first JPEG/WebP/video request dlopens the pinned decoder from the image's
`/opt/tensorfold/lib`; if one refuses by version or is missing, the server's log line names it
(and `TENSORFOLD_LIBTURBOJPEG` / `TENSORFOLD_LIBWEBP` can point at another copy in a pinch) —
PNG needs no external library.

### Receipts captured (2026-10-19 window, first v1.1.0 bring-up)

Marker: `tf=v1.1.0@76aafcf274e60a5e31a59ce145b2f71f7b366e9d image=glm53-native-tf:v1.1.0
model=nvidia/GLM-5.3-Flash-NVFP4@da920bb0b9f4a06727223a349e55468e38352348 worker=giles@192.168.2.2
context=40960 parallel=4 vision=1 head=192.168.2.1`. (Cite the marker, never `--version`: the
binary prints the in-tree manifest default "1.0.12"; upstream injects release numbers only at
`zig build dist` packaging.)

**Bring-up.** Both containers started 16:35:40Z; `/health` up 16:52:15Z — **16 min 35 s** total:
the byte cross-check read 190.4 GiB in **717.7 s (rank 0) / 718.0 s (rank 1)**, `0/33 cached`
(first pass — the digest-cache volume was empty; restarts read no shard); then ~4.6 min of
weights + tower + admission. An earlier first attempt died at family open with
`LibraryUnavailable` (the tower's cuBLASLt dlopen — fixed by bundling `libcublasLt.so.13`;
see docs/LAUNCH.md §7). Rank 1 notes `RoCE all-gather: VerbsUnavailable (); NCCL carries every
gather` — the minimal image ships no libibverbs; NCCL's path is the one served (as in v1.0.12).

**Startup admission + per-rank memory (rank 0's line):**
`model 94.78 GiB; 4 streams at once, 0.83 GiB each at a 40960-token window, of 4.3 GiB left
after a 12.2 GiB reserve; prompts in passes of up to 2048 rows, a 1.8 GiB prompt cache`.
Against the v1.0.12 baseline (2026-10-16): model **93.72 → 94.78 GiB** (+1.06 GiB — the bf16
vision tower, unquantized, on rank 0); streams 0.89 → 0.83 GiB (upstream's index rings, #984);
prompt cache 3.9 → 1.8 GiB; free 6.5 → 4.3 GiB. `/health`: active 101.77 GB; **peak after the
image gates 104.58 GB** (vs v1.0.12's 100.6 GB at the same window/stream count — +1.2 GB for
the tower, +2.8 GB transient during media work).

**Image gates — 6/6 pass** (scripts/media_gates.py, data-URL PNGs, temperature 0): drafted=plain
(per-request `"draft": false`) identical; changed-image cache isolation clean on both a red and
a blue square (turn 1 answers 'Red'/'Blue', turn 2 recalls its own conversation's colour, and
the two images' turn-1 replies differ); resumed=fresh past an image identical; 4 concurrent
image requests all equal their solo replies.

**Video gates — 3/3 pass (9/9 with the image gates, after the video ran):** the clip answered
correctly ('A static television test pattern is displayed, with a small digital display on the
right side updating.'), the video request drafted = plain, and a repeat equalled solo. The
post-video `/health` peak is unchanged (104.58 GB — the image gates had already set it).

**Host-level memory during serving** (both hosts, all gates done): brain02 (rank 0, tower side)
105.2 GiB of ~122, brain03 (rank 1) 102.6 GiB — the ~2.6 GiB delta is the tower and the media
path on rank 0.
