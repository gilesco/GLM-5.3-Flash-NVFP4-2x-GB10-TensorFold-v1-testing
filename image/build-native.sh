#!/usr/bin/env bash
# Runs inside the build stage of image/Dockerfile: installs Zig, checks out TensorFold at the
# pinned commit, builds the CUDA fatbins (sm_121 for the GB10) and the native server, and
# collects binary + libraries into /out. Every step echoes what it runs, so a failed first
# build on new hardware is readable in the docker build log.
set -euo pipefail

ZIG_VERSION="${ZIG_VERSION:?}"
TF_REPO="${TF_REPO:?}"
TF_COMMIT="${TF_COMMIT:?}"
SM_ARCH="${SM_ARCH:-121}"

echo "== tools present =="
command -v curl || { apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils git; }
command -v git || apt-get install -y --no-install-recommends git
# Tools the media-decoder builds need below (shasum is perl; build_ffmpeg.sh verifies its
# tarball with it). The NGC base has gcc/make; install only what is missing.
_missing=""
command -v make >/dev/null || _missing="$_missing make"
command -v cmake >/dev/null || _missing="$_missing cmake"
command -v shasum >/dev/null || _missing="$_missing perl"
if [[ -n "$_missing" ]]; then apt-get update && apt-get install -y --no-install-recommends $_missing; fi
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
[[ -x "$NVCC" ]] || { echo "no nvcc in the builder image (BASE_IMAGE must be a CUDA devel image)" >&2; exit 1; }
"$NVCC" --version | tail -1
HOST_ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m)"   # aarch64 on the GB10s
case "$HOST_ARCH" in aarch64|arm64) ZIG_TARGET="aarch64-linux-gnu"; ZIG_KEY="aarch64-linux" ;;
  x86_64|amd64) ZIG_TARGET="x86_64-linux-gnu"; ZIG_KEY="x86_64-linux" ;;
  *) echo "unsupported builder arch: $HOST_ARCH" >&2; exit 1 ;; esac

echo "== zig $ZIG_VERSION =="
if ! command -v zig >/dev/null; then
  URL="$(curl -s https://ziglang.org/download/index.json \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['$ZIG_VERSION']['$ZIG_KEY']['tarball'])")"
  echo "zig from $URL"
  curl -sL "$URL" -o /tmp/zig.tar.xz
  mkdir -p /usr/local/zig && tar -xJf /tmp/zig.tar.xz -C /usr/local/zig --strip-components=1
  ln -sf /usr/local/zig/zig /usr/local/bin/zig
fi
zig version

echo "== checkout $TF_COMMIT =="
git clone "$TF_REPO" /src/TensorFold
git -C /src/TensorFold checkout --detach "$TF_COMMIT"
echo "checked out: $(git -C /src/TensorFold rev-parse HEAD)"

cd /src/TensorFold
echo "== fatbins (sm_$SM_ARCH) =="
zig build fatbins -Dtarget="$ZIG_TARGET" -Dnvcc="$NVCC" -Dsm="$SM_ARCH"
ls -la zig-out/fatbin/ || true

echo "== native server =="
# packaging/README.md: Linux targets use baseline CPU features; -Dsm/-Dnvcc select the GPU
# inputs. If a future build.zig renames those options, fall back to the plain target build.
zig build native -Dtarget="$ZIG_TARGET" -Dcpu=baseline -Dsm="$SM_ARCH" -Dnvcc="$NVCC" \
  || zig build native -Dtarget="$ZIG_TARGET" -Dcpu=baseline
BIN=/src/TensorFold/zig-out/native/bin/tensorfold-native
[[ -x "$BIN" ]] || { echo "build did not produce $BIN" >&2; exit 1; }
"$BIN" --version

echo "== libtfvideo (video input): upstream's own recipe =="
# tools/zig/build_ffmpeg.sh builds a static FFmpeg 9.0.2 (PyAV's decoders only) and
# `zig build video` compiles libtfvideo over it, installing to zig-out/native/lib/. Without it,
# --vision still serves images but refuses video parts at the first request.
bash tools/zig/build_ffmpeg.sh /opt/ffmpeg
zig build video -Dffmpeg=/opt/ffmpeg
ls -la zig-out/native/lib/

echo "== the pinned image decoders (core/picture/jpeg.zig, webp.zig) =="
# The server dlopens libturbojpeg.so.0 / libwebpdecoder.so.3 from ../lib beside the binary and
# refuses them by version: libjpeg-turbo 3.1.4.1 (Pillow 12.3's) and libwebp 1.6.0 (Pillow
# 13.0.0's). Ubuntu 24.04's system copies are older, so build the pins here.
mkdir -p /opt/decoders/src && cd /opt/decoders/src
curl -fsSLO https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.4.1.tar.gz
tar xf 3.1.4.1.tar.gz
cmake -S libjpeg-turbo-3.1.4.1 -B libjpeg -DCMAKE_BUILD_TYPE=Release -DENABLE_STATIC=0 -DENABLE_SHARED=1
cmake --build libjpeg -j"$(getconf _NPROCESSORS_ONLN)"
cp -L "$(find "$PWD/libjpeg" -name 'libturbojpeg.so.0*' | sort | head -1)" /opt/decoders/libturbojpeg.so.0
curl -fsSLO https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.tar.gz
tar xf v1.6.0.tar.gz
cmake -S libwebp-1.6.0 -B webp -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=1 -DWEBP_BUILD_ANIM_UTILS=0 -DWEBP_BUILD_C_WEBP=0 -DWEBP_BUILD_DWEBP=0 -DWEBP_BUILD_GIF2WEBP=0 -DWEBP_BUILD_IMG2WEBP=0 -DWEBP_BUILD_VWEBP=0 -DWEBP_BUILD_WEBPINFO=0 -DWEBP_BUILD_WEBPMUX=0 -DWEBP_BUILD_EXTRAS=0
cmake --build webp -j"$(getconf _NPROCESSORS_ONLN)"
cp -L "$(find "$PWD/webp" -name 'libwebpdecoder.so*' | sort | head -1)" /opt/decoders/libwebpdecoder.so.3
cd /src/TensorFold

echo "== collect /out =="
mkdir -p /out/bin /out/lib
cp "$BIN" /out/bin/
# The media libraries the server dlopens from ../lib at the first JPEG/WebP/video: libtfvideo
# from the video build, the two pinned decoders from above. libcuda is deliberately NOT copied —
# at run time the host's driver provides it.
if [[ -d zig-out/native/lib ]]; then cp -L zig-out/native/lib/* /out/lib/; fi
[[ -e /opt/decoders/libturbojpeg.so.0 ]] && cp /opt/decoders/libturbojpeg.so.0 /out/lib/libturbojpeg.so.0
[[ -e /opt/decoders/libwebpdecoder.so.3 ]] && cp /opt/decoders/libwebpdecoder.so.3 /out/lib/libwebpdecoder.so.3
# NCCL: the hosts have no system libnccl and the engine opens it at run time; take the
# builder's. libcuda is deliberately NOT copied — at run time the host's driver provides it.
ldd "$BIN" > /out/ldd.txt || true
NCCL_SO="$(ldconfig -p | awk '/libnccl\.so\.2/{print $NF; exit}')"
if [[ -n "$NCCL_SO" && -e "$NCCL_SO" ]]; then
  cp -L "$NCCL_SO" /out/lib/libnccl.so.2
  echo "bundled $(readlink -f "$NCCL_SO") as /out/lib/libnccl.so.2"
else
  echo "WARNING: no libnccl.so.2 found in the builder image; rank startup over hosts will fail until one is bundled" >&2
  [[ -n "${ALLOW_NO_NCCL:-}" ]] || exit 1
fi
echo "== /out ready =="
ls -laR /out
