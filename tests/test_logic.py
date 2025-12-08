"""
Unit Testing of the application's logic
"""
import pytest
from pathlib import Path
from PIL import Image
from mylib.classifier import PetClassifier, predict


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary sample image for testing."""
    img = Image.new('RGB', (224, 224), color='blue')
    img_path = tmp_path / "sample.jpg"
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def classifier():
    """Create a classifier instance for testing."""
    return PetClassifier()

def test_classifier_initialization(classifier):
    """Test that classifier initializes correctly."""
    assert classifier.session is not None
    assert classifier.input_name is not None
    assert len(classifier.class_labels) > 0

def test_preprocess(sample_image_path, classifier):
    """Test image preprocessing."""
    preprocessed = classifier.preprocess(sample_image_path)
    assert preprocessed.shape == (1, 3, 224, 224)

def test_predict_with_file_path(sample_image_path, classifier):
    """Test predict function with file path."""
    result = classifier.predict(sample_image_path)
    assert result in classifier.class_labels

def test_predict_with_pil_image(classifier):
    """Test predict function with PIL Image."""
    img = Image.new('RGB', (224, 224), color='green')
    result = classifier.predict(img)
    assert result in classifier.class_labels

def test_predict_file_not_found(classifier):
    """Test predict with non-existent file."""
    with pytest.raises(FileNotFoundError):
        classifier.predict("nonexistent.jpg")

