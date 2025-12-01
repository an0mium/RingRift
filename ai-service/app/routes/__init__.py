"""FastAPI route modules for the AI service."""

from .replay import router as replay_router

__all__ = ["replay_router"]
