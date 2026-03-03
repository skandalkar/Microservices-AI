"""
FastAPI backend for serving ML model predictions
"""

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from models.ml_model.agri_price_predictor_ml import agriPricePredictor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from models.ml_model import agriPricePredictor

app = FastAPI(
    title="Agricultural Price Prediction API",
    description="AI-powered API for predicting agricultural product prices",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = agriPricePredictor()
        model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'ml_model',
            'agri_price_model.pkl'
        )
        predictor.load_model(model_path)

        print("Model loaded successfully at startup")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        print("Run: python ml_model/agri_price_predictor_ml.py")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")


class PredictionInput(BaseModel):
    product: str = Field(
        ...,
        description="Agricultural product",
        example="wheat"
    )
    quality: str = Field(
        ...,
        description="Quality grade: low, medium, high",
        example="high"
    )
    season: str = Field(
        ...,
        description="Season: rabi, kharif, zaid",
        example="rabi"
    )
    state: str = Field(
        ...,
        description="State: maharashtra, punjab, up, mp, haryana, rajasthan, karnataka",
        example="punjab"
    )
    quantity: float = Field(
        ...,
        description="Quantity in quintals",
        example=50.0
    )
    market_distance: float = Field(
        ...,
        description="Distance to nearest market in km",
        example=25.0
    )
    organic_certified: int = Field(
        default=0,
        description="Is product organic certified? 0=No, 1=Yes",
        example=0
    )

    class Config:
        schema_extra = {
            "example": {
                "product": "wheat",
                "quality": "high",
                "season": "rabi",
                "state": "punjab",
                "quantity": 50,
                "market_distance": 25,
                "organic_certified": 1
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for price prediction"""
    price_per_quintal: float = Field(
        ...,
        description="Predicted price per quintal in INR"
    )
    min_price: float = Field(
        ...,
        description="Minimum expected price (confidence interval)"
    )
    max_price: float = Field(
        ...,
        description="Maximum expected price (confidence interval)"
    )
    confidence: float = Field(
        ...,
        description="Confidence score as percentage (0-100)"
    )


class PredictionResponse(BaseModel):
    success: bool
    message: str
    prediction: Optional[PredictionOutput] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    return {
        "name": "Agricultural Price Prediction API",
        "version": "1.0.0",
        "description": "AI-powered price estimation for Indian agricultural products",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "products": "/products"
        }
    }


@app.get("/health")
async def health_check():
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. API not ready."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "API is running and model is ready"
    }


@app.get("/products")
async def get_products():
    Note = {
        "These are dummy prices for products. It will serve real-time price after post deployment and CI/CD pipelines integration."}
    products = {
        "wheat": "2,100/quintal (base price)",
        "rice": "2,800/quintal (base price)",
        "cotton": "6,200/quintal (base price)",
        "sugarcane": "310/quintal (base price)",
        "soybean": "4,500/quintal (base price)",
        "tomato": "2,200/quintal (base price)",
        "onion": "1,800/quintal (base price)",
        "chana": "5,200/quintal (base price - Grams/Chickpea)",
        "tur": "6,800/quintal (base price - Pigeon Pea)",
        "jowar": "3,200/quintal (base price - Sorghum)"
    }
    qualities = ["low", "medium", "high"]
    seasons = ["rabi", "kharif", "zaid"]
    states = ["maharashtra", "punjab", "up", "mp", "haryana", "rajasthan", "karnataka"]

    return {
        "Note": Note,
        "products": products,
        "qualities": qualities,
        "seasons": seasons,
        "states": states,
        "count": {
            "products": len(products),
            "qualities": len(qualities),
            "seasons": len(seasons),
            "states": len(states)
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(input_data: PredictionInput):
    # Check if model is loaded
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Validate input dataset
        if input_data.quantity <= 0:
            raise ValueError("Quantity must be greater than 0")

        if input_data.market_distance < 0:
            raise ValueError("Market distance cannot be negative")

        # Convert Pydantic model to dictionary
        input_dict = input_data.dict()

        # Make prediction
        result = predictor.predict(input_dict)

        # Return successful response
        return PredictionResponse(
            success=True,
            message="Price prediction successful",
            prediction=PredictionOutput(
                price_per_quintal=result['price_per_quintal'],
                min_price=result['min_price'],
                max_price=result['max_price'],
                confidence=result['confidence']
            )
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(ve)}"
        )
    except KeyError as ke:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input field: {str(ke)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(inputs: list[PredictionInput]):
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        results = []
        for input_data in inputs:
            input_dict = input_data.dict()
            result = predictor.predict(input_dict)
            results.append({
                "input": input_dict,
                "prediction": result
            })

        return {
            "success": True,
            "count": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "model_type": "Random Forest Regressor",
        "n_estimators": predictor.model.n_estimators,
        "max_depth": predictor.model.max_depth,
        "feature_names": predictor.feature_names,
        "encoders": list(predictor.encoders.keys()),
        "training_samples": "5000+",
        "expected_accuracy": "R² Score: 0.85-0.92",
        "mean_absolute_error": "±₹150-250"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(exc):
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(ValueError)
async def value_error_handler(exc):
    return {
        "success": False,
        "error": f"Validation error: {str(exc)}",
        "status_code": 400
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5001)
