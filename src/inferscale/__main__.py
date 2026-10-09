"""Entry point: ``python -m inferscale`` or ``inferscale-gateway``."""

from __future__ import annotations

import logging

import uvicorn

from inferscale.app import create_app
from inferscale.config import Settings


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        # One worker per pod: scale out with replicas (HPA), not processes, so
        # admission limits and metrics stay per-process and accurate.
        workers=1,
        timeout_keep_alive=75,
        # On SIGTERM, finish in-flight streams (within the pod's grace period).
        timeout_graceful_shutdown=45,
    )


if __name__ == "__main__":
    main()
