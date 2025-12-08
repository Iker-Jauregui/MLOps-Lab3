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

def test_prediction_deterministic(client, sample_image_bytes):
    """Test that same image produces same prediction."""
    # Make first prediction
    response1 = client.post(
        "/predict",
        files={"file": ("prediction1.jpg", sample_image_bytes, "image/jpeg")}
    )
    prediction1 = response1.json()["predicted_class"]
    
    # Make second prediction with same image
    response2 = client.post(
        "/predict",
        files={"file": ("prediction2.jpg", sample_image_bytes, "image/jpeg")}
    )
    prediction2 = response2.json()["predicted_class"]
    
    assert prediction1 == prediction2


def test_different_image_sizes(client):
    """Test that the API handles different image sizes correctly."""
    sizes = [(100, 100), (224, 224), (500, 500), (1920, 1080)]
    
    for width, height in sizes:
        # Create image of specific size
        img = Image.new('RGB', (width, height), color=(100, 100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            "/predict",
            files={"file": (f"test_{width}x{height}.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data


def test_different_image_formats(client):
    """Test that the API handles different image formats (JPEG, PNG, TIFF)."""
    formats = ['JPEG', 'PNG', 'TIFF']
    
    for fmt in formats:
        img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=fmt)
        img_bytes.seek(0)
        
        extension = fmt.lower()
        mime_type = f"image/{extension}"
        
        response = client.post(
            "/predict",
            files={"file": (f"test.{extension}", img_bytes, mime_type)}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data


def test_grayscale_image_conversion(client):
    """Test that grayscale images are converted to RGB correctly."""
    # Create grayscale image
    img = Image.new('L', (224, 224), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data


def test_rgba_image_conversion(client):
    """Test that RGBA images are converted to RGB correctly."""
    # Create RGBA image (with alpha channel)
    img = Image.new('RGBA', (224, 224), color=(100, 100, 100, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
