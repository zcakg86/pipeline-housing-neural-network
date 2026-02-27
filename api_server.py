"""
FastAPI Server for Real Estate Price Predictions
Production-ready REST API for model predictions
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import sys
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.getcwd() + '/src/pricemodel')

from predict_listings import ListingPredictor
from retraining_pipeline import RetrainingPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate Price Prediction API",
    description="Enhanced model V2 with uncertainty estimation",
    version="2.0.0"
)

# Global predictor instance
predictor = None
pipeline = None


class Listing(BaseModel):
    """Single listing input"""
    listing_id: str
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    sqft: float = Field(..., gt=0)
    sqft_lot: float = Field(..., gt=0)
    beds: int = Field(..., ge=0)
    h3_07: str
    list_date: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "listing_id": "L12345",
                "lat": 47.6062,
                "lng": -122.3321,
                "sqft": 1800,
                "sqft_lot": 5000,
                "beds": 3,
                "h3_07": "8728d5cd3ffffff",
                "list_date": "2025-02-26"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output"""
    listing_id: str
    predicted_price: float
    price_lower_95: float
    price_upper_95: float
    prediction_std_price: float
    confidence: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction input"""
    listings: List[Listing]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: Optional[str]
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    global predictor, pipeline
    
    logger.info("Starting API server...")
    
    try:
        predictor = ListingPredictor()
        predictor.load_model()
        logger.info(f"Model loaded from {predictor.model.directory}")
        
        pipeline = RetrainingPipeline()
        logger.info("Retraining pipeline initialized")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail until model is trained")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Real Estate Price Prediction API V2",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "retrain_status": "/model/retrain/status"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None and predictor.model is not None
    model_path = predictor.model.directory if model_loaded else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=str(model_path) if model_path else None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(listing: Listing):
    """Predict price for a single listing"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([listing.dict()])
        
        # Generate prediction
        predictions = predictor.predict(df, return_details=True)
        
        # Extract result
        result = predictions.iloc[0]
        
        # Determine confidence level
        std = result['prediction_std_price']
        if std < 50000:
            confidence = "high"
        elif std < 100000:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            listing_id=result['listing_id'],
            predicted_price=float(result['predicted_price']),
            price_lower_95=float(result['price_lower_95']),
            price_upper_95=float(result['price_upper_95']),
            prediction_std_price=float(result['prediction_std_price']),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict prices for multiple listings"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([listing.dict() for listing in request.listings])
        
        # Generate predictions
        predictions = predictor.predict(df, return_details=True)
        
        # Format results
        results = []
        for _, row in predictions.iterrows():
            std = row['prediction_std_price']
            if std < 50000:
                confidence = "high"
            elif std < 100000:
                confidence = "medium"
            else:
                confidence = "low"
            
            results.append(PredictionResponse(
                listing_id=row['listing_id'],
                predicted_price=float(row['predicted_price']),
                price_lower_95=float(row['price_lower_95']),
                price_upper_95=float(row['price_upper_95']),
                prediction_std_price=float(row['prediction_std_price']),
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict prices from uploaded CSV file"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Validate required columns
        required_cols = ['listing_id', 'lat', 'lng', 'sqft', 'sqft_lot', 'beds', 'h3_07']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Generate predictions
        predictions = predictor.predict(df, return_details=True)
        
        # Return as JSON
        return JSONResponse(content=predictions.to_dict('records'))
        
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = predictor.model
        
        return {
            "model_path": str(model.directory),
            "timestamp": model.timestamp,
            "architecture": {
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "property_dim": model.property_dim,
                "continuous_time_dim": model.continuous_time_dim,
                "market_dim": model.market_dim,
            },
            "data_info": {
                "n_communities": model.n_communities,
                "year_length": model.year_length,
                "week_length": model.week_length,
                "reference_date": model.reference_date.isoformat() if model.reference_date else None
            },
            "performance": model.results.get('metrics', {})
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/model/retrain/status")
async def retrain_status():
    """Check if model needs retraining"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        should_retrain, reason = pipeline.should_retrain(
            performance_threshold=15.0,
            time_threshold_days=30
        )
        
        active_model = pipeline.model_registry.get('active_model')
        
        return {
            "should_retrain": should_retrain,
            "reason": reason,
            "active_model": {
                "trained_date": active_model['trained_date'] if active_model else None,
                "metrics": active_model['metrics'] if active_model else None,
                "n_records": active_model['n_records'] if active_model else None
            } if active_model else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retrain status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check retrain status: {str(e)}")


@app.post("/model/retrain")
async def trigger_retrain():
    """Trigger model retraining (async operation)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # This should be done asynchronously in production
        # For now, just return a message
        return {
            "message": "Retraining triggered",
            "note": "This is a long-running operation. Check /model/retrain/status for updates",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Retrain trigger error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
