# FramePipe CPU image: builds the pinned dependencies and FramePipe, runs the
# full CPU test suite (the build fails if any test fails), then ships a small
# runtime image containing only the CLI and the OpenCV libraries it links.
#
#   docker build -t framepipe .
#   docker run --rm framepipe doctor
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" framepipe process \
#       --input in.png --output out.png --config /opt/framepipe/configs/edges_720p.json
#
# CUDA is not included: GPU execution needs a host driver and the CUDA preset.
#
# BASE selects the distribution for both stages (default Ubuntu 24.04 / GCC 13;
# debian:bookworm / GCC 12 is used for the terminal demo, see docs/demo.tape).

ARG BASE=ubuntu:24.04
FROM ${BASE} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
ENV FRAMEPIPE_DEPS=/opt/deps PATH=/venv/bin:$PATH

# Layer 1: Python tooling and C++ deps (cached unless their pins change).
COPY requirements.txt ./
RUN python3 -m venv /venv && pip install --no-cache-dir -r requirements.txt
COPY scripts/install_deps.sh scripts/
RUN scripts/install_deps.sh "$FRAMEPIPE_DEPS" && rm -rf /tmp/framepipe-deps-build

# Layer 2: FramePipe itself; tests gate the image.
COPY . .
RUN cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu

FROM ${BASE} AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/deps/lib/libopencv_core.so.410 \
                  /opt/deps/lib/libopencv_imgproc.so.410 \
                  /opt/deps/lib/libopencv_imgcodecs.so.410 /usr/local/lib/
COPY --from=build /src/build/cpu/bin/framepipe /usr/local/bin/framepipe
COPY --from=build /src/configs /opt/framepipe/configs
COPY --from=build /src/tests/fixtures /opt/framepipe/fixtures
RUN ldconfig && useradd --create-home --uid 1001 framepipe
USER framepipe
WORKDIR /work
ENTRYPOINT ["framepipe"]
CMD ["doctor"]
