# main.py

"""
MindCare AI - Mental Health Analysis API
Main application file with FastAPI server configuration
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import router
import uvicorn
import sys
import time

# Create FastAPI app with metadata
app = FastAPI(
    title="MindCare AI - Mental Health Analysis API",
    description="""
    ## 🧠 MindCare AI API
    
    Multimodal mental health monitoring using AI-powered analysis of:
    * 📝 **Text** - Emotion classification and sentiment analysis
    * 🎤 **Speech** - Voice pattern and acoustic analysis (coming soon)
    * 📹 **Video** - Facial expression recognition (coming soon)
    
    ### Features
    * Emotion Detection (Joy, Sadness, Anger, Anxiety, Neutral)
    * Sentiment Analysis (-1 to +1 scale)
    * Linguistic Feature Extraction
    * Wellness Score Calculation (0-10)
    * Crisis Indicators Detection
    
    ### Technology Stack
    * **Model**: DistilBERT (fine-tuned on 43K emotion examples)
    * **Accuracy**: ~77% on 5-emotion classification
    * **Processing**: <1 second per text analysis
    
    ### Social Impact
    Providing free mental health monitoring for individuals who cannot afford 
    professional psychiatric care (₹10,000-30,000 per therapy course in India).
    
    ---
    
    **Project by:** Gupta Kruti Narendra (22C22005)  
    **Institution:** B.Tech CSE (AI) Final Year Project  
    **Credits:** 14
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "MindCare AI Team",
        "email": "support@mindcare-ai.com",
    },
    license_info={
        "name": "Educational Use Only",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# CORS middleware - allows requests from different origins (mobile app, web, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://your-app-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request details
    print(f"📡 {request.method} {request.url.path} - "
          f"Status: {response.status_code} - "
          f"Time: {process_time:.3f}s")
    
    return response

# Include API routes with prefix
app.include_router(
    router, 
    prefix="/api/v1", 
    tags=["Text Analysis"]
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run tasks when application starts"""
    print("\n" + "="*70)
    print("🚀 MindCare AI API - Starting Up...")
    print("="*70)
    
    # Print Python version
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    
    # Try to pre-load model
    try:
        print("\n📦 Pre-loading AI model...")
        from models.model_loader import get_text_model
        model = get_text_model()
        print("✅ Model loaded successfully!")
        print(f"   Type: DistilBERT")
        print(f"   Parameters: {model.model.num_parameters():,}")
        print(f"   Device: {model.device}")
        print(f"   Emotions: {list(model.id_to_label.values())}")
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load model")
        print(f"   Reason: {str(e)}")
        print(f"   Model will be loaded on first API request")
    
    print("\n📚 API Documentation:")
    print("   Swagger UI:  http://localhost:8000/docs")
    print("   ReDoc:       http://localhost:8000/redoc")
    print("   OpenAPI:     http://localhost:8000/openapi.json")
    
    print("\n🔗 Available Endpoints:")
    print("   POST /api/v1/analyze-text")
    print("   POST /api/v1/batch-analyze")
    print("   GET  /api/v1/health")
    print("   GET  /api/v1/model-info")
    
    print("\n" + "="*70)
    print("✅ Server Ready! Listening for requests...")
    print("="*70 + "\n")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run cleanup tasks when application shuts down"""
    print("\n" + "="*70)
    print("🛑 MindCare AI API - Shutting Down...")
    print("="*70)
    print("✅ Cleanup completed")
    print("👋 Goodbye!\n")

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information and quick links
    
    Returns basic information about the API and available endpoints.
    """
    return {
        "message": "Welcome to MindCare AI - Mental Health Analysis API",
        "version": "1.0.0",
        "status": "running",
        "project": {
            "name": "MindCare AI",
            "description": "Multimodal mental health monitoring using AI",
            "author": "Gupta Kruti Narendra (22C22005)",
            "institution": "B.Tech CSE (AI) Final Year Project"
        },
        "endpoints": {
            "analyze_text": {
                "path": "/api/v1/analyze-text",
                "method": "POST",
                "description": "Analyze text for emotion and mental health indicators"
            },
            "batch_analyze": {
                "path": "/api/v1/batch-analyze",
                "method": "POST",
                "description": "Analyze multiple texts at once (max 50)"
            },
            "health_check": {
                "path": "/api/v1/health",
                "method": "GET",
                "description": "Check API health and model status"
            },
            "model_info": {
                "path": "/api/v1/model-info",
                "method": "GET",
                "description": "Get information about the AI model"
            }
        },
        "documentation": {
            "interactive_docs": "/docs",
            "alternative_docs": "/redoc",
            "openapi_schema": "/openapi.json"
        },
        "features": [
            "Text emotion classification (5 emotions)",
            "Sentiment analysis (-1 to +1)",
            "Linguistic feature extraction",
            "Wellness score calculation (0-10)",
            "Crisis indicators detection",
            "Batch processing support"
        ],
        "model_info": {
            "architecture": "DistilBERT-base-uncased",
            "parameters": "66M",
            "emotions": ["joy", "sadness", "anger", "anxiety", "neutral"],
            "accuracy": "~77%",
            "training_samples": "43,410"
        }
    }

# Health check endpoint (simple, no dependencies)
@app.get("/health", tags=["Health"])
async def simple_health():
    """
    Simple health check without loading model
    
    Useful for container orchestration and monitoring tools.
    """
    return {
        "status": "healthy",
        "service": "MindCare AI API",
        "version": "1.0.0",
        "uptime": "running"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    
    Catches any error that wasn't handled by route-specific handlers
    and returns a consistent error response.
    """
    print(f"❌ Unhandled exception: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "message": "An unexpected error occurred. Please try again or contact support."
        }
    )

# 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 Not Found errors"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": f"The endpoint '{request.url.path}' does not exist.",
            "suggestion": "Visit /docs for available endpoints"
        }
    )

# Main entry point
if __name__ == "__main__":
    """
    Run the server directly with: python main.py
    
    For production deployment, use a process manager like:
    - gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
    - uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    
    print("\n🔧 Starting Uvicorn server...")
    print("💡 Press CTRL+C to stop the server\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",      # Listen on all network interfaces
        port=8000,            # Port number
        reload=True,          # Auto-reload on code changes (development only)
        log_level="info",     # Logging level
        access_log=True,      # Log all requests
    )