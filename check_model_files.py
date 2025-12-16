# check_model_files.py
import json
import os

model_path = "models/text_model_final"

print("🔍 Checking model files...\n")

json_files = ['config.json', 'tokenizer_config.json', 'label_mappings.json']

for json_file in json_files:
    filepath = os.path.join(model_path, json_file)
    
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"📄 {json_file}:")
        print(f"   Size: {size:,} bytes")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
                    print(f"   ✅ Valid JSON with {len(data)} keys")
                else:
                    print(f"   ❌ File is EMPTY!")
        except json.JSONDecodeError as e:
            print(f"   ❌ INVALID JSON: {e}")
            print(f"   First 100 chars: {content[:100]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"❌ {json_file} NOT FOUND!")
    
    print()