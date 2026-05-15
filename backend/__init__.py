"""
Backend API for Digital Twin dashboard.

This module provides FastAPI endpoints for the digital twin visualization,
connecting the Python dynamics simulation to the HTML frontend.
"""

from .app import app

__all__ = ["app"]
