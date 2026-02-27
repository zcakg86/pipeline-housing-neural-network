"""
API Test Script
Quick test to verify the API is working correctly
"""
import requests
import json
import sys
from datetime import datetime

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"  ✓ Status: {data['status']}")
        print(f"  ✓ Model loaded: {data['model_loaded']}")
        if data.get('model_path'):
            print(f"  ✓ Model path: {data['model_path']}")
        return True
    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to API. Is it running?")
        print("    Start with: make api")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_root():
    """Test root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        print(f"  ✓ API: {data['message']}")
        print(f"  ✓ Version: {data['version']}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        
        if response.status_code == 503:
            print("  ⚠️  Model not loaded. Train a model first:")
            print("    make train")
            return False
        
        response.raise_for_status()
        data = response.json()
        
        print(f"  ✓ Model path: {data['model_path']}")
        print(f"  ✓ Architecture: {data['architecture']['embedding_dim']}D embeddings")
        print(f"  ✓ Communities: {data['data_info']['n_communities']}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_prediction():
    """Test single prediction"""
    print("\nTesting prediction endpoint...")
    
    listing = {
        "listing_id": "TEST001",
        "lat": 47.6062,
        "lng": -122.3321,
        "sqft": 1800,
        "sqft_lot": 5000,
        "beds": 3,
        "h3_07": "8728d5cd3ffffff",
        "list_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=listing,
            timeout=10
        )
        
        if response.status_code == 503:
            print("  ⚠️  Model not loaded. Train a model first:")
            print("    make train")
            return False
        
        response.raise_for_status()
        data = response.json()
        
        print(f"  ✓ Listing ID: {data['listing_id']}")
        print(f"  ✓ Predicted Price: ${data['predicted_price']:,.0f}")
        print(f"  ✓ 95% CI: ${data['price_lower_95']:,.0f} - ${data['price_upper_95']:,.0f}")
        print(f"  ✓ Confidence: {data['confidence']}")
        print(f"  ✓ Std Dev: ${data['prediction_std_price']:,.0f}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction"""
    print("\nTesting batch prediction endpoint...")
    
    listings = [
        {
            "listing_id": "BATCH001",
            "lat": 47.6062,
            "lng": -122.3321,
            "sqft": 1800,
            "sqft_lot": 5000,
            "beds": 3,
            "h3_07": "8728d5cd3ffffff",
            "list_date": datetime.now().strftime("%Y-%m-%d")
        },
        {
            "listing_id": "BATCH002",
            "lat": 47.6205,
            "lng": -122.3493,
            "sqft": 2200,
            "sqft_lot": 6500,
            "beds": 4,
            "h3_07": "8728d540bffffff",
            "list_date": datetime.now().strftime("%Y-%m-%d")
        }
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"listings": listings},
            timeout=15
        )
        
        if response.status_code == 503:
            print("  ⚠️  Model not loaded. Train a model first:")
            print("    make train")
            return False
        
        response.raise_for_status()
        data = response.json()
        
        print(f"  ✓ Processed {len(data)} listings")
        for pred in data:
            print(f"    - {pred['listing_id']}: ${pred['predicted_price']:,.0f} ({pred['confidence']} confidence)")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_retrain_status():
    """Test retraining status endpoint"""
    print("\nTesting retraining status endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/retrain/status", timeout=5)
        
        if response.status_code == 503:
            print("  ⚠️  Pipeline not initialized")
            return False
        
        response.raise_for_status()
        data = response.json()
        
        print(f"  ✓ Should retrain: {data['should_retrain']}")
        print(f"  ✓ Reason: {data['reason']}")
        
        if data.get('active_model'):
            active = data['active_model']
            if active.get('trained_date'):
                print(f"  ✓ Last trained: {active['trained_date']}")
            if active.get('metrics'):
                print(f"  ✓ Model error: {active['metrics'].get('mean_abs_pct_error', 'N/A'):.2f}%")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("API TEST SUITE")
    print("="*80)
    print()
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Model Info", test_model_info),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Retraining Status", test_retrain_status),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    elif passed > 0:
        print("\n⚠️  Some tests failed. Check the output above.")
        print("\nCommon issues:")
        print("  - API not running: make api")
        print("  - Model not trained: make train")
        print("  - Wrong port: check docker-compose.yml")
    else:
        print("\n❌ All tests failed. Is the API running?")
        print("\nStart the API with:")
        print("  make api")
        print("\nOr check if it's running:")
        print("  docker-compose ps")
    
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
