"""FastAPI application for image classification."""

import io
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from mylib.classifier import predict as predict_func, get_classifier

# Create an instance of FastAPI
app = FastAPI(
    title="API of the Pet Classifier using FastAPI",
    description="API to perform image predictions using mylib.classifier",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")

# Startup event to preload the model
@app.on_event("startup")
async def startup_event():
    """Load the model at startup to avoid delays on first request."""
    get_classifier()
    print("Model loaded successfully at startup")

# Initial endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home endpoint returning HTML template."""
    return templates.TemplateResponse(request=request, name="home.html")


# Main endpoint to perform the image prediction
@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...)
):
    """
    Predict the class of the input image.

    Parameters
    ----------
    file : UploadFile
        Image file to classify

    Returns
    -------
    dict
        Dictionary with predicted class
    """
    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Get prediction
        prediction = predict_func(image)

        return {"predicted_class": prediction}

    except (FileNotFoundError, IOError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing image: {str(e)}"
        ) from e


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
