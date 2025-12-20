# download_model.py - Working version for 1GB+ RAM

import os
import gdown
import zipfile
import shutil

def download_and_extract_model():
    """Download model from Google Drive"""
    
    model_dir = "models/text_model_final"
    
    # Check if model already exists
    if os.path.exists(f"{model_dir}/pytorch_model.bin") or \
       os.path.exists(f"{model_dir}/model.safetensors"):
        print("✅ Model already exists, skipping download")
        return
    
    print("📥 Downloading trained model from Google Drive...")
    
    # YOUR GOOGLE DRIVE FILE ID
    GOOGLE_DRIVE_FILE_ID = "1JZZuGp8QpjJg_ZOKLOd2gQZos0zDNROU"  # From your logs
    
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download"
    output = "model_files.zip"
    
    try:
        # Download with gdown
        print("Downloading...")
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        print("📦 Extracting...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            # Extract to temp location first
            zip_ref.extractall("temp_model/")
        
        # Move files to correct location
        # Check if extracted to subfolder
        extracted_path = "temp_model/text_model_final"
        if os.path.exists(extracted_path):
            # Move from subfolder
            for item in os.listdir(extracted_path):
                shutil.move(
                    os.path.join(extracted_path, item),
                    os.path.join(model_dir, item)
                )
        else:
            # Move from root
            for item in os.listdir("temp_model/"):
                shutil.move(
                    os.path.join("temp_model/", item),
                    os.path.join(model_dir, item)
                )
        
        # Cleanup
        shutil.rmtree("temp_model/")
        os.remove(output)
        
        print("✅ Model downloaded successfully!")
        
        # Verify
        if os.path.exists(f"{model_dir}/pytorch_model.bin"):
            print("✅ pytorch_model.bin verified!")
            size = os.path.getsize(f"{model_dir}/pytorch_model.bin") / (1024*1024)
            print(f"   Size: {size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTrying fallback: fresh DistilBERT...")
        download_fresh_model()

def download_fresh_model():
    """Fallback: Download fresh DistilBERT"""
    print("📥 Downloading fresh DistilBERT (untrained)...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import json
        
        model_dir = "models/text_model_final"
        os.makedirs(model_dir, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=5,
            id2label={0: 'joy', 1: 'sadness', 2: 'anger', 3: 'anxiety', 4: 'neutral'},
            label2id={'joy': 0, 'sadness': 1, 'anger': 2, 'anxiety': 3, 'neutral': 4}
        )
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Label mappings
        with open(f'{model_dir}/label_mappings.json', 'w') as f:
            json.dump({
                'id_to_label': {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'anxiety', 4: 'neutral'},
                'label_to_id': {'joy': 0, 'sadness': 1, 'anger': 2, 'anxiety': 3, 'neutral': 4}
            }, f, indent=2)
        
        print("✅ Fallback model ready!")
        print("⚠️  Note: This is untrained - predictions will be random")
        
    except Exception as e:
        print(f"❌ Fallback also failed: {e}")
        raise

if __name__ == "__main__":
    download_and_extract_model()