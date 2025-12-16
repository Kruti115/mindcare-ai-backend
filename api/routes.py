# api/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from models.model_loader import get_text_model
from typing import Dict, Any, Optional
import time

router = APIRouter()

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    user_id: Optional[str] = Field(None, description="Optional user ID for tracking")

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[Any, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@router.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(input_data: TextInput):
    """
    Analyze text for emotion, sentiment, and mental health indicators
    
    **Example Request:**
```json
    {
        "text": "I feel so hopeless and worthless today",
        "user_id": "user123"
    }
```
    
    **Example Response:**
```json
    {
        "success": true,
        "data": {
            "emotion": {
                "primary": "sadness",
                "confidence": 0.82
            },
            "sentiment": {"compound": -0.75},
            "wellness_score": 3.2,
            "interpretation": "You seem to be experiencing sadness..."
        },
        "processing_time": 0.15
    }
```
    """
    start_time = time.time()
    
    try:
        # Validate input
        text = input_data.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Get model
        model = get_text_model()
        
        # Analyze
        result = model.analyze(text)
        
        # Add metadata
        result['user_id'] = input_data.user_id
        result['timestamp'] = time.time()
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            data=result,
            processing_time=round(processing_time, 3)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        return AnalysisResponse(
            success=False,
            error=str(e),
            processing_time=round(processing_time, 3)
        )

@router.get("/health")
async def health_check():
    """
    Check if API is running and model is loaded
    
    **Returns:**
```json
    {
        "status": "healthy",
        "service": "MindCare AI - Text Analysis",
        "version": "1.0.0",
        "model_loaded": true
    }
```
    """
    try:
        model = get_text_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "service": "MindCare AI - Text Analysis",
        "version": "1.0.0",
        "model_loaded": model_loaded
    }

@router.get("/model-info")
async def model_info():
    """
    Get information about the loaded model
    
    **Returns:**
```json
    {
        "model_type": "DistilBERT",
        "emotions": ["joy", "sadness", "anger", "anxiety", "neutral"],
        "device": "cpu",
        "parameters": 66955010
    }
```
    """
    try:
        model = get_text_model()
        return {
            "model_type": "DistilBERT",
            "model_name": "distilbert-base-uncased",
            "emotions": list(model.id_to_label.values()),
            "num_emotions": len(model.id_to_label),
            "device": str(model.device),
            "parameters": model.model.num_parameters(),
            "max_length": 128
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze(texts: list[str]):
    """
    Analyze multiple texts at once
    
    **Example Request:**
```json
    ["I'm so happy!", "I feel sad", "This is neutral"]
```
    """
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")
    
    try:
        model = get_text_model()
        results = []
        
        for text in texts:
            if text.strip():
                result = model.analyze(text)
                results.append(result)
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))