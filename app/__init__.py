# ABOUTME: Flask application factory - creates and configures the app
# ABOUTME: Registers blueprints, initializes extensions, sets up error handlers

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask

from app.config import config
from app.extensions import db, login_manager, migrate, csrf

# Configure logging to stdout for production (Railway, etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_name: Configuration name (development, production, testing)

    Returns:
        Configured Flask application
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    logger.info(f"Creating app with config: {config_name}")

    app = Flask(__name__, template_folder="interfaces/web/templates")
    app.config.from_object(config[config_name])

    logger.info(f"Database URI configured: {bool(app.config.get('SQLALCHEMY_DATABASE_URI'))}")

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)

    # Configure login manager
    login_manager.login_view = "web.login"
    login_manager.login_message_category = "info"

    # Register blueprints
    from app.interfaces.web import bp as web_bp
    app.register_blueprint(web_bp)

    from app.interfaces.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    csrf.exempt(api_bp)  # API uses token auth, not CSRF

    # Create database tables
    logger.info("Creating database tables...")
    with app.app_context():
        # Import models to register them
        from app.infrastructure.database import models  # noqa: F401
        db.create_all()
    logger.info("Database tables created successfully")

    # Setup file logging (optional - skip if can't write to filesystem)
    if not app.debug and not app.testing:
        try:
            if not os.path.exists("logs"):
                os.mkdir("logs")
            file_handler = RotatingFileHandler(
                "logs/palette_transfer.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10,
            )
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
        except OSError:
            logger.warning("Could not set up file logging - using stdout only")
        app.logger.setLevel(logging.INFO)
        app.logger.info("Palette Transfer startup")

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return {"error": "Internal server error"}, 500

    # Health check
    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app
