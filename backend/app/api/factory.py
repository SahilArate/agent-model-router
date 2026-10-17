"""Application factory. Tests pass their own Services; production builds the defaults."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.api.services import Services, build_services

# Origins of the local React dev server.
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(title="Agent Model Router", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.services = services or build_services()
    app.include_router(router)
    return app