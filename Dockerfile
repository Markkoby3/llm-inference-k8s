# syntax=docker/dockerfile:1.7
# Gateway image. The GPU engines use their upstream images (vllm/vllm-openai and
# nvcr.io/nvidia/tritonserver); this image only runs the lightweight router.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
COPY --from=build /dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin inferscale
USER 10001
EXPOSE 8080
ENTRYPOINT ["inferscale-gateway"]
