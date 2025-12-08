"""
Image classifier library
"""

import random
from pathlib import Path
from PIL import Image


def predict(image_path, class_names=None):
    """
    Predict the class of an image.

    Loads image from file or PIL Image object and returns predicted class.

    Parameters
    ----------
    image_path : str, Path, or PIL.Image
        Path to image file (str or Path object) or PIL Image object directly.
        Supported formats: JPG, PNG, BMP, GIF, TIFF.
    class_names : list of str, optional
        List of class names. Default: ['cardboard', 'paper', 'plastic', 'metal', 'trash', 'glass']

    Returns
    -------
    str
        Predicted class name (randomly selected from class_names).

    Raises
    ------
    FileNotFoundError
        If image file path does not exist.
    IOError
        If image file cannot be read.
    ValueError
        If image format is not supported or class_names is empty.

    Examples
    --------
    >>> predicted_class = predict("sample.jpg", ['cat', 'dog'])
    """
    if class_names is None:
        class_names = ["cardboard", "paper", "plastic", "metal", "trash", "glass"]

    if not class_names:
        raise ValueError("class_names cannot be empty")

    try:
        # Handle both file paths and PIL Images
        if isinstance(image_path, (str, Path)):
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image_path type: {type(image_path)}")

        # For Lab1: randomly select a class
        predicted_class = random.choice(class_names)
        return predicted_class

    except FileNotFoundError:
        raise
    except Exception as e:
        raise IOError(f"Error loading image: {str(e)}") from e
