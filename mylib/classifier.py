"""
Image classifier library
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from torchvision.models import MobileNet_V2_Weights


class PetClassifier:
    """
    ONNX-based pet classifier for image classification.
    
    This classifier loads an ONNX model and class labels once during initialization,
    then can be used for multiple predictions efficiently.
    """
    
    def __init__(self, model_path=None, labels_path=None):
        """
        Initialize the classifier with ONNX model and class labels.
        
        Parameters
        ----------
        model_path : str or Path, optional
            Path to the ONNX model file. If None, uses default path relative to project root.
        labels_path : str or Path, optional
            Path to the JSON file containing class labels. If None, uses default path.
        """
        # Get project root directory (parent of mylib)
        project_root = Path(__file__).parent.parent
        
        # Set default paths relative to project root
        if model_path is None:
            model_path = project_root / "production_model" / "mobilenetv2_pet_classifier_run8e0704.onnx"
        if labels_path is None:
            labels_path = project_root / "production_model" / "mobilenetv2_pet_classifier_run8e0704_class_labels.json"

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        
        # Start ONNX Runtime session
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Get session input name
        self.input_name = self.session.get_inputs()[0].name
        
        # Load class labels from JSON
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_labels_dict = json.load(f)
            # Convert keys to integers and sort by index
            self.class_labels = [class_labels_dict[str(i)] for i in range(len(class_labels_dict))]
        
        # Define preprocessing transforms
        self.transform = self.transform = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    
    def preprocess(self, image):
        """
        Preprocess image for model inference.
        
        Parameters
        ----------
        image : PIL.Image or str or Path
            Input image as PIL Image object or path to image file
            
        Returns
        -------
        numpy.ndarray
            Preprocessed image array ready for inference
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        input_tensor = self.transform(image)
        
        # Add batch dimension and convert to numpy
        input_array = input_tensor.unsqueeze(0).numpy()
        
        return input_array
    
    def predict(self, image_path):
        """
        Predict the class of an image.
        
        Parameters
        ----------
        image_path : str, Path, or PIL.Image
            Path to image file or PIL Image object
            
        Returns
        -------
        str
            Predicted class name
            
        Raises
        ------
        FileNotFoundError
            If image file path does not exist
        IOError
            If image file cannot be read
        ValueError
            If image format is not supported
            
        Examples
        --------
        >>> classifier = PetClassifier()
        >>> predicted_class = classifier.predict("dog.jpg")
        """
        try:
            # Preprocess the image
            input_array = self.preprocess(image_path)
            
            # Create inputs dictionary
            inputs = {self.input_name: input_array}
            
            # Run inference
            outputs = self.session.run(None, inputs)
            
            # Get logits (first output)
            logits = outputs[0]
            
            # Get predicted class index
            pred_idx = int(np.argmax(logits, axis=1)[0])
            
            # Get class label
            predicted_class = self.class_labels[pred_idx]
            
            return predicted_class
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise IOError(f"Error during prediction: {str(e)}") from e


# Create a global instance for the API to use
_classifier_instance = None

def get_classifier():
    """
    Get or create the global classifier instance.
    
    This ensures the model is loaded only once and reused across predictions.
    
    Returns
    -------
    PetClassifier
        Global classifier instance
    """
    # pylint: disable=global-statement
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = PetClassifier()
    return _classifier_instance


def predict(image_path):
    """
    Predict the class of an image using the global classifier instance.
    
    This function maintains backward compatibility with the original API
    while using the ONNX model for predictions.
    
    Parameters
    ----------
    image_path : str, Path, or PIL.Image
        Path to image file or PIL Image object
        
    Returns
    -------
    str
        Predicted class name
        
    Examples
    --------
    >>> predicted_class = predict("sample.jpg")
    """
    classifier = get_classifier()
    return classifier.predict(image_path)