# ABOUTME: REST API interface - JSON endpoints for programmatic access
# ABOUTME: Thin controllers that delegate to services

from flask import Blueprint

bp = Blueprint("api", __name__)

from app.interfaces.api import routes  # noqa: E402, F401
