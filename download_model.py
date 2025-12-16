# download_model.py

import os
import gdown
import zipfile

def download_and_extract_model():
    """Download model from Google Drive"""
    
    model_dir = "models/text_model_final"
    
    # Check if model already exists
    if os.path.exists(f"{model_dir}/config.json") and \
       os.path.exists(f"{model_dir}/pytorch_model.bin"):
        print("✅ Model already exists, skipping download")
        return
    
    print("📥 Downloading model from Google Drive...")
    
    # Your Google Drive shareable link
    # Format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    # Extract FILE_ID and use: https://drive.google.com/uc?id=FILE_ID
    
    GOOGLE_DRIVE_FILE_ID = "1crauuXzU8VFzUATVtNJV71pIqJ6wFAbm"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    output = "model_files.zip"
    
    try:
        gdown.download(url, output, quiet=False)
        
        print("📦 Extracting model files...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall("models/")
        
        os.remove(output)
        print("✅ Model downloaded and extracted successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("Using fallback: downloading fresh DistilBERT...")
        
        # Fallback: download fresh model
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=5
        )
        
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        print("✅ Fallback model downloaded!")

if __name__ == "__main__":
    download_and_extract_model()

## https://drive.google.com/file/d/1crauuXzU8VFzUATVtNJV71pIqJ6wFAbm/view?usp=sharing