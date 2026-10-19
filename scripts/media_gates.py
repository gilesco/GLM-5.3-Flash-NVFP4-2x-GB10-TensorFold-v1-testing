#!/usr/bin/env python3
"""Media receipts for ashhart/TensorFold#971 on the two-GB10 NVFP4 window.

Runs the gates upstream's 2026-10-19 comment owes — drafted/plain, changed-image cache
isolation, resumed/fresh, concurrent/solo, and (with --video) the video gates — against the
running server and prints a receipt block to paste into the issue. Standard library only.

Requires the server started with --vision (this recipe's default). Media goes in as OpenAI-style
image_url / video_url parts with data: base64 URLs, so no --vision-urls and no external files
are needed except --video. Drafted vs plain uses the per-request "draft": false body field, so
nothing is restarted. Temperature 0 everywhere; equality is on the full reply content.

    python3 scripts/media_gates.py                          # against 127.0.0.1:8000
    python3 scripts/media_gates.py --video /path/clip.mp4   # adds the video gates
"""

import argparse
import base64
import json
import struct
import sys
import threading
import urllib.request
import zlib

COLORS = {"red": (224, 32, 32), "blue": (32, 64, 224)}
COLOR_QUESTION = ("The image contains a single solid-color square. What color is it? "
                  "Answer with only the color name.")
RECALL_QUESTION = "What color did you say the square was? Answer with only the color name."


def png_solid(rgb, size=64):
    """A minimal valid PNG of one solid color (stdlib only)."""
    w = h = size
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


class Client:
    def __init__(self, base_url, model, timeout):
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, draft=None):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 1024}
        if draft is not None:
            body["draft"] = draft
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise SystemExit(f"HTTP {e.code} from {self.base}: {detail}\n"
                             "  (a refusal naming --vision means the server was started without it)")
        return out["choices"][0]["message"]["content"]

    def health(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - the receipt notes it either way
            return {"error": repr(e)}


def user_img(url, text):
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": text},
    ]}


def user_text(text):
    return {"role": "user", "content": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    ap.add_argument("--video", help="path to a clip (h264/hevc/vp9 in mp4/mkv/webm) for the video gates")
    ap.add_argument("--timeout", type=int, default=600, help="per-request seconds")
    args = ap.parse_args()
    c = Client(args.base_url, args.model, args.timeout)
    urls = {name: data_url(png_solid(rgb)) for name, rgb in COLORS.items()}
    results, receipt = [], []

    def record(name, ok, detail):
        results.append(ok)
        line = ("PASS" if ok else "FAIL") + f"  {name}: {detail}"
        print(("  ✓ " if ok else "  ✗ ") + f"{name}: {detail}")
        receipt.append(line)

    # 1. drafted = plain (an image request, per-request draft toggle)
    req = [user_img(urls["red"], COLOR_QUESTION)]
    drafted = c.chat(req, draft=True)
    plain = c.chat(req, draft=False)
    record("drafted=plain (image)", drafted == plain,
           "identical" if drafted == plain else f"draft={drafted!r} plain={plain!r}")

    # 2. changed-image cache isolation: same shape, swapped image; each conversation's
    #    turns must be about its own image, not a stale cached prefix.
    convs = {}
    for name in ("red", "blue"):
        first = c.chat([user_img(urls[name], COLOR_QUESTION)])
        second = c.chat([user_img(urls[name], COLOR_QUESTION), {"role": "assistant", "content": first},
                         user_text(RECALL_QUESTION)])
        convs[name] = (first, second)
        ok = name in first.lower() and name in second.lower()
        record(f"changed-image isolation ({name})", ok, f"turn1={first!r} turn2={second!r}")
    distinct = convs["red"][0] != convs["blue"][0]
    record("changed-image isolation (distinct images)", distinct,
           "turn-1 replies differ per image" if distinct else
           f"identical replies for different images: {convs['red'][0]!r}")

    # 3. resumed = fresh past an image: the continued turn 2 vs the same turn 2 in one
    #    fresh request carrying the whole history (and the image) again.
    red_first, red_second = convs["red"]
    fresh = c.chat([user_img(urls["red"], COLOR_QUESTION),
                    {"role": "assistant", "content": red_first}, user_text(RECALL_QUESTION)])
    record("resumed=fresh past an image", red_second == fresh,
           "identical" if red_second == fresh else f"resumed={red_second!r} fresh={fresh!r}")

    # 4. concurrent = solo: the same four image requests fired together, then alone.
    qs = ["What color is the square? Answer with only the color name.",
          "Reply with the single word: red or blue.",
          "Is the square red or blue? One word.",
          "Name the square's color in one word."]
    reqs = [[user_img(urls["red"], q)] for q in qs]
    solo = [c.chat(r) for r in reqs]
    conc = [None] * len(reqs)
    def run(i):
        conc[i] = c.chat(reqs[i])
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(reqs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = conc == solo
    record("concurrent=solo (images)", ok,
           "all 4 equal their solo replies" if ok else
           f"mismatches: {[i for i in range(len(qs)) if conc[i] != solo[i]]}")

    # 5. video, if a clip was given: works, drafted=plain, concurrent=solo.
    if args.video:
        with open(args.video, "rb") as f:
            vurl = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
        vq = "Describe the motion in this clip in one sentence."
        vreq = [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": vurl}},
            {"type": "text", "text": vq}]}]
        vreply = c.chat(vreq)
        record("video request answered", bool(vreply and vreply.strip()), repr(vreply)[:160])
        vd = c.chat(vreq, draft=True)
        vp = c.chat(vreq, draft=False)
        record("video drafted=plain", vd == vp, "identical" if vd == vp else f"{vd!r} vs {vp!r}")
        v2 = c.chat(vreq)
        record("video repeated=solo", v2 == vreply, "identical" if v2 == vreply else f"{v2!r} vs {vreply!r}")

    health = c.health()
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passed = sum(results)
    print(f"\n{passed}/{len(results)} gates passed")
    print("\nPaste into the issue:\n")
    print("```text")
    print(f"#971 media receipts — {stamp} — {args.base_url} model={args.model}")
    for line in receipt:
        print(line)
    mem = health.get("memory", health)
    print(f"health.memory: {json.dumps(mem, sort_keys=True)[:1200]}")
    print("```")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
