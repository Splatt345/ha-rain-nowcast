# THE BASE IMAGE IS FIXED, AND DELIBERATELY NOT `ARG BUILD_FROM`.
#
# The Supervisor passes --build-arg BUILD_FROM=ghcr.io/home-assistant/base:latest
# on every build. That base is Alpine/musl, and this add-on cannot run on it:
# opencv-python-headless and onnxruntime publish no musl wheels, so pip would
# fall back to building OpenCV and ONNX Runtime from source inside the add-on
# build. Declaring `ARG BUILD_FROM` let that value win over the default and the
# build died on `apt-get: not found`.
#
# Not declaring the ARG makes the Supervisor's value an unused build arg - a
# warning, not an error - and the base stays Debian whatever it passes. This is
# also why there is no build.yaml: the Supervisor rejected its `build_from`
# ("does not match regular expression", because a bare `python:3.12-slim` has no
# namespace/ prefix) and then told us build.yaml is deprecated anyway.
#
# pysteps publishes no wheel for this Python, so its Cython extensions
# (_proesmans, _vet) get compiled from source. That needs a toolchain, which
# has no business being in the shipped image - hence the separate build stage.
FROM python:3.12-slim-bookworm AS builder
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# NOTHING IS PRUNED HERE, AND THAT IS A MEASURED DECISION. Both of the usual
# image-slimming steps were tried on this dependency set and both shipped a
# broken image:
#
#   strip --strip-unneeded on the .so files  -> numpy cannot load its compiled
#     extensions and fails with the misleading "you should not try to import
#     numpy from its source directory". The scientific wheels need symbols
#     strip considers unneeded.
#
#   deleting bundled tests/ directories (62 MB) -> ModuleNotFoundError:
#     numpy._core.tests. It is imported for real at runtime, not just by tests.
#
# __pycache__ (82 MB) is kept on purpose as well: deleting it only moves the
# cost to container start, where Python writes it straight back into the
# writable layer, so the host saves nothing and every restart pays for it.
#
# The real saving on this image was the dependency choice, not pruning:
# onnxruntime instead of PyTorch took it from 2.06 GB to 1.08 GB.

FROM python:3.12-slim-bookworm

# libgomp is the one runtime piece the compiled bits need (built with -fopenmp).
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY nowcast /app/nowcast
COPY web /app/web
# Trained SmaAt-UNet weights (~16 MB). Absent -> server falls back to STEPS.
COPY models /app/models

WORKDIR /app
# MALLOC_ARENA_MAX caps glibc's per-thread allocation arenas. It creates up to
# 8x ncores of them for a threaded program and never returns them to the OS,
# which for this workload is pure waste: measured 784 MiB resident by default
# against 571 MiB at 2, for one environment variable. The usual cost is
# allocator contention under heavy concurrency, and this forecasts once every
# five minutes.
ENV PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    RADAR_CACHE_DIR=/data/cache

EXPOSE 8061
CMD ["python", "-m", "nowcast.server"]

ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
ARG BUILD_VERSION

LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION} \
    org.opencontainers.image.title="${BUILD_NAME}" \
    org.opencontainers.image.description="${BUILD_DESCRIPTION}" \
    org.opencontainers.image.source="https://github.com/${BUILD_REPOSITORY}" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}
