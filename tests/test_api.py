"""
Integration testing with the API
"""
import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from api.api import app


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample image in memory for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_home_endpoint(client):
    """Verify that the endpoint / returns the right message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict(client, sample_image_bytes):
    """Verify that the endpoint /predict performs the class prediction correctly."""
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data


def test_predict_invalid_file(client):
    """Verify that the endpoint /predict manages correctly invalid files."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
