# -*- coding: utf-8 -*-
"""
Flask extensions - initialized here to avoid circular imports.

Import these in models.py, routes, and anywhere else needed.
Initialize with app in __init__.py via init_app().
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect

# Database
db = SQLAlchemy()

# Authentication
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Database migrations
migrate = Migrate()

# CSRF protection
csrf = CSRFProtect()
