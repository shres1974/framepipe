#!/usr/bin/env bash
# Build FramePipe's pinned C++ dependencies from source into a user prefix.
# No root access required. Requires: a C++17 compiler, curl, and cmake/ninja
# (the project venv provides pinned cmake 3.30.5 and ninja 1.11.1.1).
#
# Usage: scripts/install_deps.sh [PREFIX]
#   PREFIX defaults to $HOME/.local/framepipe-deps
#
# Afterwards configure with -DCMAKE_PREFIX_PATH=PREFIX (presets do this
# automatically through the FRAMEPIPE_DEPS environment variable).
set -euo pipefail

PREFIX="${1:-$HOME/.local/framepipe-deps}"
WORK="${FRAMEPIPE_DEPS_WORK:-/tmp/framepipe-deps-build}"
JOBS="$(nproc)"

OPENCV_VERSION=4.10.0
OPENCV_SHA256=b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9
GTEST_VERSION=1.15.2
GTEST_SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
JSON_VERSION=3.11.3
JSON_SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406

mkdir -p "$WORK" "$PREFIX"
cd "$WORK"

fetch() { # url sha256 out
  if [[ ! -f "$3" ]]; then curl -fsSL "$1" -o "$3"; fi
  echo "$2  $3" | sha256sum -c -
}

fetch "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" "$OPENCV_SHA256" opencv.tar.gz
fetch "https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz" "$GTEST_SHA256" gtest.tar.gz
fetch "https://github.com/nlohmann/json/archive/refs/tags/v${JSON_VERSION}.tar.gz" "$JSON_SHA256" json.tar.gz

for t in opencv gtest json; do rm -rf "$t-src"; mkdir "$t-src"; tar -xzf "$t.tar.gz" -C "$t-src" --strip-components=1; done

cmake -S json-src -B json-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DJSON_BuildTests=OFF
cmake --build json-build && cmake --install json-build

cmake -S gtest-src -B gtest-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_GMOCK=OFF
cmake --build gtest-build -j "$JOBS" && cmake --install gtest-build

# Minimal OpenCV: core, imgproc (labeled baseline only), imgcodecs with
# bundled PNG/JPEG/zlib so no system image libraries are needed.
cmake -S opencv-src -B opencv-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_LIST=core,imgproc,imgcodecs \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF \
  -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON \
  -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF \
  -DWITH_OPENEXR=OFF -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF \
  -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF -DWITH_AVIF=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_GTK=OFF \
  -DWITH_QT=OFF -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF \
  -DWITH_ITT=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_PROTOBUF=OFF \
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DOPENCV_GENERATE_PKGCONFIG=ON
cmake --build opencv-build -j "$JOBS" && cmake --install opencv-build

echo "Dependencies installed in $PREFIX"
