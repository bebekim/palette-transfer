# ABOUTME: Integration tests for Google OAuth authentication
# ABOUTME: Tests OAuth routes with mocked Google responses

import pytest
from unittest.mock import patch, MagicMock

from app import create_app
from app.extensions import db
from app.infrastructure.database.models import UserModel


@pytest.fixture
def app():
    """Create test Flask app with OAuth configured."""
    app = create_app("testing")
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["GOOGLE_CLIENT_ID"] = "test-client-id"
    app.config["GOOGLE_CLIENT_SECRET"] = "test-client-secret"

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def existing_user(app):
    """Create an existing user in the database."""
    with app.app_context():
        user = UserModel(
            email="existing@example.com",
            display_name="Existing User",
        )
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
        return user.id


class TestGoogleLoginRoute:
    """Test /login/google route."""

    def test_login_google_redirects_to_google(self, client):
        """Test login/google redirects to Google OAuth."""
        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_redirect.return_value = "redirect_response"

            client.get("/login/google")

            mock_oauth.create_client.assert_called_with("google")
            mock_google.authorize_redirect.assert_called_once()

    def test_login_google_redirects_when_authenticated(self, client, app, existing_user):
        """Test login/google redirects to dashboard if already logged in."""
        with client.session_transaction() as sess:
            sess["_user_id"] = str(existing_user)

        response = client.get("/login/google")

        assert response.status_code == 302
        assert "/dashboard" in response.location

    def test_login_google_shows_error_when_not_configured(self, client, app):
        """Test login/google shows error when OAuth not configured."""
        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = None

            response = client.get("/login/google", follow_redirects=True)

            assert b"Google login is not configured" in response.data


class TestGoogleCallbackRoute:
    """Test /login/google/callback route."""

    def test_callback_creates_new_user(self, client, app):
        """Test callback creates new user from Google info."""
        mock_token = {
            "userinfo": {
                "email": "newuser@gmail.com",
                "sub": "google-id-123",
                "name": "New User",
                "picture": "https://example.com/avatar.jpg",
            }
        }

        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_access_token.return_value = mock_token

            response = client.get("/login/google/callback", follow_redirects=True)

            assert response.status_code == 200

            with app.app_context():
                user = UserModel.query.filter_by(email="newuser@gmail.com").first()
                assert user is not None
                assert user.oauth_provider == "google"
                assert user.oauth_id == "google-id-123"
                assert user.display_name == "New User"
                assert user.avatar_url == "https://example.com/avatar.jpg"
                assert user.is_verified is True
                assert user.password_hash is None

    def test_callback_links_existing_user(self, client, app, existing_user):
        """Test callback links OAuth to existing user by email."""
        mock_token = {
            "userinfo": {
                "email": "existing@example.com",
                "sub": "google-id-456",
                "name": "Google Name",
                "picture": "https://example.com/google-avatar.jpg",
            }
        }

        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_access_token.return_value = mock_token

            response = client.get("/login/google/callback", follow_redirects=True)

            assert response.status_code == 200

            with app.app_context():
                user = UserModel.query.filter_by(email="existing@example.com").first()
                assert user is not None
                assert user.oauth_provider == "google"
                assert user.oauth_id == "google-id-456"
                # Original name preserved since user already had display_name
                assert user.display_name == "Existing User"
                # Password should still be set
                assert user.password_hash is not None

    def test_callback_handles_oauth_error(self, client, app):
        """Test callback handles OAuth error gracefully."""
        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_access_token.side_effect = Exception("OAuth failed")

            response = client.get("/login/google/callback", follow_redirects=True)

            assert b"Failed to authenticate with Google" in response.data

    def test_callback_handles_missing_email(self, client, app):
        """Test callback handles missing email from Google."""
        mock_token = {
            "userinfo": {
                "sub": "google-id-789",
                "name": "No Email User",
            }
        }

        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_access_token.return_value = mock_token

            response = client.get("/login/google/callback", follow_redirects=True)

            assert b"Could not retrieve email from Google" in response.data

    def test_callback_rejects_deactivated_user(self, client, app):
        """Test callback rejects deactivated users."""
        with app.app_context():
            user = UserModel(
                email="deactivated@example.com",
                oauth_provider="google",
                oauth_id="google-deactivated",
                is_active=False,
            )
            db.session.add(user)
            db.session.commit()

        mock_token = {
            "userinfo": {
                "email": "deactivated@example.com",
                "sub": "google-deactivated",
            }
        }

        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_access_token.return_value = mock_token

            response = client.get("/login/google/callback", follow_redirects=True)

            assert b"account has been deactivated" in response.data

    def test_callback_updates_last_login(self, client, app, existing_user):
        """Test callback updates last_login timestamp."""
        mock_token = {
            "userinfo": {
                "email": "existing@example.com",
                "sub": "google-id-login",
            }
        }

        with patch("app.interfaces.web.routes.oauth") as mock_oauth:
            mock_google = MagicMock()
            mock_oauth.create_client.return_value = mock_google
            mock_google.authorize_access_token.return_value = mock_token

            client.get("/login/google/callback", follow_redirects=True)

            with app.app_context():
                user = UserModel.query.filter_by(email="existing@example.com").first()
                assert user.last_login is not None


class TestOAuthUserModel:
    """Test UserModel OAuth fields."""

    def test_oauth_user_without_password(self, app):
        """Test creating OAuth-only user without password."""
        with app.app_context():
            user = UserModel(
                email="oauth-only@example.com",
                oauth_provider="google",
                oauth_id="google-oauth-only",
            )
            db.session.add(user)
            db.session.commit()

            fetched = UserModel.query.filter_by(email="oauth-only@example.com").first()
            assert fetched.password_hash is None
            assert fetched.oauth_provider == "google"
            assert fetched.oauth_id == "google-oauth-only"

    def test_user_can_have_both_password_and_oauth(self, app):
        """Test user can have both password and OAuth linked."""
        with app.app_context():
            user = UserModel(
                email="both@example.com",
                oauth_provider="google",
                oauth_id="google-both",
            )
            user.set_password("securepassword")
            db.session.add(user)
            db.session.commit()

            fetched = UserModel.query.filter_by(email="both@example.com").first()
            assert fetched.password_hash is not None
            assert fetched.check_password("securepassword")
            assert fetched.oauth_provider == "google"
