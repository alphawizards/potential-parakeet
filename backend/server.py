"""
Server Entry Point
==================
Compatibility layer for supervisor which expects server:app.
Imports the main FastAPI application.
"""

from backend.main import app

# Re-export for uvicorn
__all__ = ["app"]
