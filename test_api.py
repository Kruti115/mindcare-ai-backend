## **FILE 11: test_api.py**
import requests
import json
import time

API_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        print("✅ Health check passed!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Health check failed: {response.status_code}")
    print()

def test_model_info():
    """Test model info endpoint"""
    print("ℹ️ Testing model info endpoint...")
    response = requests.get(f"{API_URL}/model-info")
    
    if response.status_code == 200:
        print("✅ Model info retrieved!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Model info failed: {response.status_code}")
    print()

def test_text_analysis():
    """Test text analysis endpoint"""
    test_cases = [
        "I'm so happy and excited about my new job!",
        "I feel so hopeless and worthless, nobody cares about me",
        "The weather is nice today",
        "I'm really angry and frustrated with everything!",
        "I'm worried and anxious about tomorrow"
    ]
    
    print("🧪 TESTING TEXT ANALYSIS API")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Input: '{text}'")
        
        try:
            start = time.time()
            response = requests.post(
                f"{API_URL}/analyze-text",
                json={"text": text, "user_id": "test_user"},
                timeout=10
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                
                if result['success']:
                    data = result['data']
                    print(f"✅ Emotion: {data['emotion']['primary'].upper()} "
                          f"({data['emotion']['confidence']:.1%} confidence)")
                    print(f"   Sentiment: {data['sentiment']['compound']:.2f}")
                    print(f"   Wellness: {data['wellness_score']}/10")
                    print(f"   Processing: {elapsed:.2f}s")
                    print(f"   → {data['interpretation']}")
                else:
                    print(f"❌ API Error: {result['error']}")
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Is the server running?")
            print("   Start server with: python main.py")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 80)

def test_batch_analysis():
    """Test batch analysis"""
    print("\n📦 Testing batch analysis...")
    
    texts = [
        "I'm happy!",
        "I'm sad.",
        "I'm angry!",
        "I'm worried.",
        "Just a normal day."
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/batch-analyze",
            json=texts,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analyzed {result['count']} texts successfully!")
            
            for i, r in enumerate(result['results'], 1):
                print(f"  {i}. {r['emotion']['primary']} (score: {r['wellness_score']})")
        else:
            print(f"❌ Batch analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    print()

def main():
    print("\n" + "="*80)
    print("🧪 MINDCARE AI API TEST SUITE")
    print("="*80 + "\n")
    
    # Test each endpoint
    test_health()
    test_model_info()
    test_text_analysis()
    test_batch_analysis()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()