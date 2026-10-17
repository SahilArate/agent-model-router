"""ASGI entry point:  uvicorn app.main:app --reload"""

from app.api.factory import create_app
from app.core.logging import configure_logging

configure_logging()
app = create_app()