# ABOUTME: Web interface - HTML views for browser-based access
# ABOUTME: Renders templates and handles form submissions

from flask import Blueprint

bp = Blueprint("web", __name__)

from app.interfaces.web import routes  # noqa: E402, F401
