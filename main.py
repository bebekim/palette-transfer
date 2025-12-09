# ABOUTME: Entry point for Railway deployment
# ABOUTME: Uses gunicorn to serve the Flask application in production

import os


def main():
    """Run the Flask application with gunicorn."""
    from app import create_app

    app = create_app()
    port = int(os.environ.get("PORT", 8080))

    # In production, use gunicorn
    if os.environ.get("FLASK_ENV") == "production":
        import gunicorn.app.base

        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            "bind": f"0.0.0.0:{port}",
            "workers": 2,
            "timeout": 120,
        }
        StandaloneApplication(app, options).run()
    else:
        # Development mode
        app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
