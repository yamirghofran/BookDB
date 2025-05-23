import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os # To potentially control torch threads

# --- Configuration ---
# Set threads for torch to avoid potential issues in some environments
# You might adjust this based on your CPU cores
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

MODEL_NAME = "DT12the/distilbert-sentiment-analysis"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
# Load model and tokenizer globally on startup
try:
    logger.info(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval() # Set model to evaluation mode
    logger.info("Model and tokenizer loaded successfully.")
    # Store labels for easy access
    id2label = model.config.id2label
    label2id = model.config.label2id
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    # Exit if model loading fails critically
    raise SystemExit(f"Could not load model: {e}")

# --- API Definition ---
app = FastAPI(
    title="Sentiment Analysis Service",
    description="Analyzes sentiment of input text using DistilBERT.",
)

# Define request body structure
class SentimentRequest(BaseModel):
    text: str

# Define response body structure
class SentimentResponse(BaseModel):
    text: str
    predicted_label: str
    probabilities: dict[str, float] # Dictionary mapping label name to probability

# --- Inference Function ---
def predict_sentiment(text: str) -> tuple[str, dict[str, float]]:
    """Performs sentiment analysis on the input text."""
    if not text or not isinstance(text, str):
        # Basic validation
        raise ValueError("Input text cannot be empty and must be a string.")

    try:
        # 1. Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512) # Added max_length

        # 2. Predict (disable gradients for inference)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 3. Process Output
        probabilities_tensor = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities_tensor, dim=-1).item() # Get single index
        predicted_label = id2label[predicted_class_id]

        # Create probability dictionary mapping label names to scores
        probabilities_list = probabilities_tensor.squeeze().tolist() # Convert tensor to list
        probabilities_dict = {id2label[i]: prob for i, prob in enumerate(probabilities_list)}

        return predicted_label, probabilities_dict

    except Exception as e:
        logger.error(f"Error during prediction for text '{text[:50]}...': {e}")
        # Re-raise or handle specific exceptions as needed
        raise RuntimeError(f"Prediction failed: {e}")


# --- API Endpoint ---
@app.post("/predict/", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """
    Predicts the sentiment of the provided text.

    - **text**: The input string to analyze.
    """
    logger.info(f"Received prediction request for text: '{request.text[:50]}...'")
    try:
        predicted_label, probabilities = predict_sentiment(request.text)
        logger.info(f"Prediction successful: Label='{predicted_label}'")
        return SentimentResponse(
            text=request.text,
            predicted_label=predicted_label,
            probabilities=probabilities
        )
    except ValueError as ve: # Handle validation errors specifically
        logger.warning(f"Invalid input: {ve}")
        # You might want to return a 400 Bad Request here using FastAPI's HTTPException
        # from fastapi import HTTPException
        # raise HTTPException(status_code=400, detail=str(ve))
        # For simplicity, re-raising as internal error for now
        raise RuntimeError(f"Invalid input: {ve}")
    except Exception as e:
        # Catch other potential errors during prediction
        # Consider returning a 500 Internal Server Error
        # from fastapi import HTTPException
        # raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
        raise RuntimeError(f"Internal server error during prediction: {e}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# To run the service (explained below): uvicorn sentiment_service:app --reload --port 8000