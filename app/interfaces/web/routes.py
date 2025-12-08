# ABOUTME: Web routes - HTML views for browser interface
# ABOUTME: Handles page rendering and form-based interactions

from datetime import datetime

from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, current_user, login_required

from app.interfaces.web import bp
from app.interfaces.api.deps import get_user_service
from app.infrastructure.database.repositories import SQLUserRepository
from app.infrastructure.database.models import UserModel
from app.domain.user import UserCreate
from app.domain.exceptions import ValidationError


@bp.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@bp.route("/dashboard")
@login_required
def dashboard():
    """User dashboard with transfer UI."""
    return render_template("dashboard.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    """User login page."""
    if current_user.is_authenticated:
        return redirect(url_for("web.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        # Use repository directly for auth (need password_hash access)
        repo = SQLUserRepository()
        user_model = repo.get_model_by_email(email)

        if user_model is None or not user_model.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("auth/login.html")

        if not user_model.is_active:
            flash("Your account has been deactivated.", "error")
            return render_template("auth/login.html")

        login_user(user_model, remember=remember)
        user_model.last_login = datetime.utcnow()

        from app.extensions import db
        db.session.commit()

        next_page = request.args.get("next")
        if not next_page or not next_page.startswith("/"):
            next_page = url_for("web.dashboard")

        return redirect(next_page)

    return render_template("auth/login.html")


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for("web.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Validation
        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("auth/signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("auth/signup.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("auth/signup.html")

        try:
            service = get_user_service()
            user_data = UserCreate(email=email, password=password)
            service.register(user_data)

            # Log in the new user
            repo = SQLUserRepository()
            user_model = repo.get_model_by_email(email)
            login_user(user_model)

            flash("Account created successfully!", "success")
            return redirect(url_for("web.dashboard"))

        except ValidationError as e:
            flash(e.message, "error")
            return render_template("auth/signup.html")

    return render_template("auth/signup.html")


@bp.route("/logout")
@login_required
def logout():
    """Log out current user."""
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("web.index"))
