# MindCare AI - Mental Health Analysis API

AI-powered mental health monitoring using text, speech, and video analysis.

## Features
- Text emotion classification (5 emotions)
- Sentiment analysis
- Wellness scoring
- Crisis detection

## Tech Stack
- FastAPI
- DistilBERT
- PyTorch
- Transformers

## Deployment
Deployed on Render.com

## Author
Gupta Kruti Narendra (22C22005)
B.Tech CSE (AI) Final Year Project
```

#### **3.4: Create Procfile for Render**

Create file: `Procfile` (no extension)
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT