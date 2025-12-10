# ABOUTME: Web routes - HTML views for browser interface
# ABOUTME: Handles page rendering and form-based interactions

from datetime import datetime

from flask import render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, current_user, login_required

from app.interfaces.web import bp
from app.interfaces.api.deps import get_user_service
from app.infrastructure.database.repositories import SQLUserRepository
from app.infrastructure.database.models import UserModel
from app.domain.user import UserCreate
from app.domain.exceptions import ValidationError
from app.extensions import db, oauth


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


@bp.route("/login/google")
def login_google():
    """Initiate Google OAuth login."""
    if current_user.is_authenticated:
        return redirect(url_for("web.dashboard"))

    google = oauth.create_client('google')
    if google is None:
        flash("Google login is not configured.", "error")
        return redirect(url_for("web.login"))

    redirect_uri = url_for("web.login_google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@bp.route("/login/google/callback")
def login_google_callback():
    """Handle Google OAuth callback."""
    google = oauth.create_client('google')
    if google is None:
        flash("Google login is not configured.", "error")
        return redirect(url_for("web.login"))

    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if user_info is None:
            user_info = google.userinfo()
    except Exception as e:
        current_app.logger.error(f"OAuth callback error: {e}")
        flash("Failed to authenticate with Google.", "error")
        return redirect(url_for("web.login"))

    email = user_info.get('email', '').lower()
    if not email:
        flash("Could not retrieve email from Google.", "error")
        return redirect(url_for("web.login"))

    google_id = user_info.get('sub')
    name = user_info.get('name')
    picture = user_info.get('picture')

    # Look up user by email (auto-link strategy)
    repo = SQLUserRepository()
    user_model = repo.get_model_by_email(email)

    if user_model is None:
        # Create new user
        user_model = UserModel(
            email=email,
            oauth_provider='google',
            oauth_id=google_id,
            display_name=name,
            avatar_url=picture,
            is_verified=True,  # Google verified their email
        )
        db.session.add(user_model)
        db.session.commit()
        flash("Account created successfully!", "success")
    else:
        # Link OAuth if not already linked
        if user_model.oauth_provider is None:
            user_model.oauth_provider = 'google'
            user_model.oauth_id = google_id
        # Update profile from Google if available
        if picture and not user_model.avatar_url:
            user_model.avatar_url = picture
        if name and not user_model.display_name:
            user_model.display_name = name

    if not user_model.is_active:
        flash("Your account has been deactivated.", "error")
        return redirect(url_for("web.login"))

    user_model.last_login = datetime.utcnow()
    db.session.commit()

    login_user(user_model, remember=True)
    return redirect(url_for("web.dashboard"))
