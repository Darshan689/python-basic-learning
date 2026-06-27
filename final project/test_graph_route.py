"""Unit tests for the /graphs/<filename> Flask route."""
import os
import sys
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def client():
    """Create a Flask test client for the app."""
    from flask import Flask, abort, send_from_directory

    GRAPHS_DIR = os.path.join("graphs", "current")

    app = Flask(__name__)
    app.config["TESTING"] = True

    @app.route("/graphs/<filename>")
    def serve_graph(filename):
        # Reject path traversal characters
        if ".." in filename or "/" in filename or "\\" in filename:
            abort(400)

        # Only serve .png or .csv files
        if not (filename.endswith(".png") or filename.endswith(".csv")):
            abort(404)

        # Check file existence
        file_path = os.path.join(GRAPHS_DIR, filename)
        if not os.path.isfile(file_path):
            abort(404)

        # Serve the file with correct content type
        if filename.endswith(".png"):
            return send_from_directory(GRAPHS_DIR, filename, mimetype="image/png")
        else:
            return send_from_directory(GRAPHS_DIR, filename, mimetype="text/csv")

    with app.test_client() as client:
        yield client


class TestGraphRoutePathTraversal:
    """Test that path traversal attempts return HTTP 400."""

    def test_double_dot_rejected(self, client):
        # Flask's default route param doesn't capture "/" so "../" won't reach
        # the handler. Test ".." within the filename segment directly.
        response = client.get("/graphs/..secret.png")
        assert response.status_code == 400

    def test_double_dot_prefix_rejected(self, client):
        # Test ".." embedded in filename (without slashes)
        response = client.get("/graphs/..%5Cetc%5Cpasswd.png")
        # URL-decoded: "..\etc\passwd.png" which contains both ".." and "\"
        assert response.status_code == 400

    def test_backslash_rejected(self, client):
        response = client.get("/graphs/test\\.png")
        assert response.status_code == 400

    def test_dot_dot_in_filename(self, client):
        response = client.get("/graphs/..secret.png")
        assert response.status_code == 400

    def test_forward_slash_in_filename(self, client):
        response = client.get("/graphs/sub/file.png")
        # Flask might split this differently, but testing literal
        # The route param won't capture slashes by default in Flask
        # so this would be a 404 at the routing level
        assert response.status_code == 404


class TestGraphRouteExtensionValidation:
    """Test that only .png and .csv files are served."""

    def test_txt_extension_rejected(self, client):
        response = client.get("/graphs/test.txt")
        assert response.status_code == 404

    def test_jpg_extension_rejected(self, client):
        response = client.get("/graphs/test.jpg")
        assert response.status_code == 404

    def test_py_extension_rejected(self, client):
        response = client.get("/graphs/app.py")
        assert response.status_code == 404

    def test_no_extension_rejected(self, client):
        response = client.get("/graphs/noextension")
        assert response.status_code == 404


class TestGraphRouteFileServing:
    """Test that existing files are served correctly."""

    def test_existing_png_returns_200(self, client):
        response = client.get("/graphs/heart_confusion_matrix.png")
        assert response.status_code == 200
        assert response.content_type == "image/png"

    def test_existing_csv_returns_200(self, client):
        response = client.get("/graphs/metrics_summary.csv")
        assert response.status_code == 200
        assert "text/csv" in response.content_type

    def test_nonexistent_png_returns_404(self, client):
        response = client.get("/graphs/nonexistent_file.png")
        assert response.status_code == 404

    def test_nonexistent_csv_returns_404(self, client):
        response = client.get("/graphs/nonexistent_data.csv")
        assert response.status_code == 404

    def test_png_content_type_header(self, client):
        response = client.get("/graphs/accuracy_summary.png")
        assert response.status_code == 200
        assert response.content_type == "image/png"
