# models/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from utils.analysis import (
    analyze_sentiment,
    extract_linguistic_features,
    calculate_wellness_score,
    get_interpretation
)

class TextAnalysisModel:
    def __init__(self, model_path="models/text_model_final"):
        """Load the trained text analysis model"""
        
        # Convert to absolute path
        model_path = os.path.abspath(model_path)
        print(f"📂 Loading model from: {model_path}")
        
        # Verify directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_path}\n"
                f"Please download the model from Google Drive."
            )
        
        # List files with sizes
        files = os.listdir(model_path)
        print(f"📄 Found {len(files)} files:")
        for f in files:
            size = os.path.getsize(os.path.join(model_path, f))
            print(f"   - {f} ({size:,} bytes)")
        
        # Verify config.json
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError("config.json not found!")
        
        # Validate config.json
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
                print(f"📝 config.json size: {len(config_content)} bytes")
                if not config_content.strip():
                    raise ValueError("config.json is empty!")
                config = json.loads(config_content)
                print(f"✅ config.json is valid with {len(config)} keys")
        except json.JSONDecodeError as e:
            print(f"❌ config.json is invalid JSON!")
            print(f"   Error: {e}")
            print(f"   First 100 chars: {config_content[:100]}")
            raise
        
        # Load label mappings
        self.id_to_label = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'anxiety', 4: 'neutral'}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        
        mappings_path = os.path.join(model_path, 'label_mappings.json')
        if os.path.exists(mappings_path):
            try:
                with open(mappings_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                    self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
                    self.label_to_id = mappings['label_to_id']
                    print(f"✅ Loaded label mappings from file")
            except Exception as e:
                print(f"⚠️  Could not load label_mappings.json: {e}")
                print("   Using default mappings")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 Using device: {self.device}")
        
        # Load tokenizer
        try:
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True  # Don't try to download
            )
            print(f"✅ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise RuntimeError(f"Tokenizer loading failed: {str(e)}")
        
        # Load model
        try:
            print("📥 Loading model...")
            
            # Try loading with explicit config
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,  # Don't download
                trust_remote_code=False  # Security
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Type: {self.model.config.model_type}")
            print(f"   Parameters: {self.model.num_parameters():,}")
            print(f"   Labels: {self.model.config.num_labels}")
            print(f"   Emotions: {list(self.id_to_label.values())}")
            
        except Exception as e:
            print(f"❌ Failed to load model!")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
            # Check if it's a file reading issue
            if "Expecting value" in str(e):
                print("\n🔍 This looks like a JSON parsing error.")
                print("   Checking all JSON files...")
                
                for json_file in ['config.json', 'tokenizer_config.json']:
                    json_path = os.path.join(model_path, json_file)
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                json.loads(content)
                                print(f"   ✅ {json_file} is valid")
                        except Exception as je:
                            print(f"   ❌ {json_file} is INVALID: {je}")
                            print(f"      First 200 chars: {content[:200]}")
            
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def predict_emotion(self, text: str) -> dict:
        """Predict emotion from text"""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract results
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        all_probs = predictions[0].cpu().numpy().tolist()
        
        return {
            'primary': self.id_to_label[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                self.id_to_label[i]: float(prob) 
                for i, prob in enumerate(all_probs)
            }
        }
    
    def analyze(self, text: str) -> dict:
        """Complete text analysis pipeline"""
        try:
            # 1. Emotion prediction
            emotion = self.predict_emotion(text)
            
            # 2. Sentiment analysis
            sentiment = analyze_sentiment(text)
            
            # 3. Linguistic features
            features = extract_linguistic_features(text)
            
            # 4. Wellness score
            wellness_score = calculate_wellness_score(
                emotion['primary'], 
                sentiment, 
                features
            )
            
            # 5. Interpretation
            interpretation = get_interpretation(
                wellness_score, 
                emotion['primary']
            )
            
            return {
                'emotion': emotion,
                'sentiment': sentiment,
                'linguistic_features': features,
                'wellness_score': float(wellness_score),
                'interpretation': interpretation,
                'input_length': len(text)
            }
        except Exception as e:
            print(f"❌ Error in analyze(): {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# Global model instance
_text_model = None

def get_text_model():
    """Get or create text model instance"""
    global _text_model
    if _text_model is None:
        _text_model = TextAnalysisModel()
    return _text_model