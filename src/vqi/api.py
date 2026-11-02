"""Inspection API: post an image, get OK/NOK with score, threshold and heatmap.

Every decision is written to the SQLite decision log before the answer is sent.

Run (config from ``VQI_SERVE_CONFIG``, default ``configs/serve/leather.yaml``):
    uvicorn --factory vqi.api:app_from_env --host 0.0.0.0 --port 8000
"""

import base64
import os
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile

from vqi.config import load_serve_config
from vqi.decision_log import Decision, DecisionLog
from vqi.imageio import decode_bytes
from vqi.inspection import Inspector, build_inspector, git_commit

DEFAULT_CONFIG = "configs/serve/leather.yaml"


def create_app(inspector: Inspector, log: DecisionLog) -> FastAPI:
    """Build the app around a ready inspector, so tests can pass a fake model."""
    app = FastAPI(title="Visual quality inspection")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check that also names the model being served."""
        return {"status": "ok", "model_version": inspector.model_version}

    @app.post("/inspect")
    def inspect(file: UploadFile) -> dict[str, Any]:
        """Inspect one uploaded image and log the decision."""
        try:
            image = decode_bytes(file.file.read())
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        result = inspector.inspect(image)
        decision = Decision(
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            file=file.filename or "",
            score=result.score,
            decision=result.decision,
            needs_review=result.needs_review,
            threshold=inspector.threshold,
            model_version=inspector.model_version,
            latency_ms=result.latency_ms,
            defect_type=result.defect_type.name if result.defect_type else None,
        )
        log.add(decision)
        return {
            **asdict(decision),
            "defect_probability": result.defect_type.probability if result.defect_type else None,
            "review_band": list(inspector.band),
            "heatmap_png_base64": base64.b64encode(result.heatmap_png).decode("ascii"),
        }

    @app.get("/review")
    def review(limit: int = 50) -> list[dict[str, Any]]:
        """Newest decisions waiting for an operator."""
        return log.review_queue(limit)

    @app.get("/daily")
    def daily() -> dict[str, list[dict[str, Any]]]:
        """Per UTC day: counts and defect rate, and NOK parts per defect type."""
        return {"days": log.daily_summary(), "defect_types": log.daily_defect_types()}

    return app


def app_from_env() -> FastAPI:
    """App for uvicorn: config file and ``VQI_*`` overrides from the environment."""
    config = load_serve_config(os.environ.get("VQI_SERVE_CONFIG", DEFAULT_CONFIG), os.environ)
    return create_app(build_inspector(config, git_commit()), DecisionLog(config.db_path))
