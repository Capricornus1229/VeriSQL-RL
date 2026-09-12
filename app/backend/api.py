"""FastAPI application for the VeriSQL Studio deployment."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, load_serve_config
from .database_registry import DatabaseRegistry, UnknownDatabaseError
from .model_client import (
    ModelUnavailableError,
    PromptTooLongError,
    VLLMClient,
)
from .query_service import QueryService
from .schemas import QueryRequest, QueryResponse


SERVE_CONFIG = load_serve_config()
FRONTEND_DIST = PROJECT_ROOT / "app/frontend/dist"


@asynccontextmanager
async def lifespan(application: FastAPI):
    registry = DatabaseRegistry.from_catalogs(
        SERVE_CONFIG["paths"]["train_schema_catalog"],
        SERVE_CONFIG["paths"]["dev_schema_catalog"],
        demo_queries=SERVE_CONFIG["paths"]["demo_queries"],
    )
    model_client = VLLMClient(SERVE_CONFIG)
    await model_client.startup()
    application.state.config = SERVE_CONFIG
    application.state.registry = registry
    application.state.model_client = model_client
    application.state.query_service = QueryService(
        registry,
        model_client,
        SERVE_CONFIG,
    )
    try:
        yield
    finally:
        await model_client.close()


app = FastAPI(title=SERVE_CONFIG["app"]["title"], lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[SERVE_CONFIG["frontend"]["dev_origin"]],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def get_health(request: Request) -> dict:
    model_ready = await request.app.state.model_client.health()
    return {
        "status": "ready" if model_ready else "model_unavailable",
        "api_ready": True,
        "model_ready": model_ready,
        "adapter_name": request.app.state.config["model"][
            "served_adapter_name"
        ],
        "grounding_ready": Path(
            request.app.state.config["paths"]["grounding_index"]
        ).is_file(),
        "database_count": len(request.app.state.registry),
        "version": "1.0",
    }


@app.get("/api/databases")
async def list_databases(request: Request) -> list[dict]:
    return request.app.state.registry.list_databases()


@app.get("/api/databases/{db_id}")
async def get_database(db_id: str, request: Request) -> dict:
    try:
        entry = request.app.state.registry.get_database(db_id)
    except UnknownDatabaseError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown database: {db_id}",
        ) from error
    return {
        "db_id": entry["db_id"],
        "split": entry["split"],
        "tables": entry["tables"],
        "foreign_keys": entry["foreign_keys"],
        "schema_text": entry["schema_text"],
    }


@app.post("/api/query", response_model=QueryResponse)
async def run_query(body: QueryRequest, request: Request) -> dict:
    try:
        return await request.app.state.query_service.query(body)
    except UnknownDatabaseError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown database: {body.db_id}",
        ) from error
    except PromptTooLongError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


if FRONTEND_DIST.is_dir():
    assets_directory = FRONTEND_DIST / "assets"
    if assets_directory.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_directory),
            name="assets",
        )

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend_fallback(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        return FileResponse(FRONTEND_DIST / "index.html")
