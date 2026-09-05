ARG BUILD_FROM=python:3.12-slim-bookworm

# pysteps publishes no wheel for this Python, so its Cython extensions
# (_proesmans, _vet) get compiled from source. That needs a toolchain, which
# has no business being in the shipped image - hence the separate build stage.
# hadolint ignore=DL3006
FROM ${BUILD_FROM} AS builder
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# hadolint ignore=DL3006
FROM ${BUILD_FROM}

# libgomp is the one runtime piece the compiled bits need (built with -fopenmp).
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY nowcast /app/nowcast
COPY web /app/web

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    RADAR_CACHE_DIR=/data/cache

EXPOSE 8099
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
