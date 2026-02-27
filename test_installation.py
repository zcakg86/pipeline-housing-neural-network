"""
Installation and Setup Test Script
Verifies all components are working correctly
"""
import sys
import os

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('sklearn', 'scikit-learn'),
        ('requests', 'requests'),
        ('joblib', 'joblib'),
    ]
    
    missing = []
    for module, package in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✓ All required packages installed\n")
    return True


def test_data_files():
    """Test that required data files exist"""
    print("Testing data files...")
    
    required_files = [
        'data/sales_202025.csv',
        'data/community_map.json',
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Missing data files: {', '.join(missing)}")
        print("These are required for training. Make sure your data is in place.")
        return False
    
    print("✓ All required data files present\n")
    return True


def test_module_imports():
    """Test that custom modules can be imported"""
    print("Testing custom modules...")
    
    sys.path.insert(0, os.getcwd() + '/src/pricemodel')
    
    modules = [
        'market_indicators',
        'embedding_model_v2',
        'model_manager_v2',
        'retraining_pipeline',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - ERROR: {str(e)[:50]}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("✓ All custom modules import successfully\n")
    return True


def test_market_indicators():
    """Test market indicator fetching"""
    print("Testing market indicator fetching...")
    
    try:
        sys.path.insert(0, os.getcwd() + '/src/pricemodel')
        from market_indicators import MarketIndicatorFetcher
        
        fetcher = MarketIndicatorFetcher()
        
        # Test mortgage rate fetching (will use cache or fallback)
        print("  Testing mortgage rate fetch...")
        mortgage_df = fetcher.fetch_mortgage_rates('2024-01-01', '2024-01-31')
        
        if len(mortgage_df) > 0:
            print(f"  ✓ Fetched {len(mortgage_df)} mortgage rate records")
        else:
            print("  ⚠️  No mortgage data (using fallback)")
        
        # Test unemployment fetching
        print("  Testing unemployment fetch...")
        unemployment_df = fetcher.fetch_unemployment_rate('2024-01-01', '2024-01-31')
        
        if len(unemployment_df) > 0:
            print(f"  ✓ Fetched {len(unemployment_df)} unemployment records")
        else:
            print("  ⚠️  No unemployment data (using fallback)")
        
        print("✓ Market indicator system working\n")
        return True
        
    except Exception as e:
        print(f"❌ Market indicator test failed: {e}\n")
        return False


def test_model_architecture():
    """Test that model can be instantiated"""
    print("Testing model architecture...")
    
    try:
        sys.path.insert(0, os.getcwd() + '/src/pricemodel')
        from embedding_model_v2 import EnhancedEmbeddingModel
        import torch
        
        # Create a small test model
        model = EnhancedEmbeddingModel(
            embedding_dim=8,
            hidden_dim=16,
            property_dim=3,
            continuous_time_dim=5,
            market_dim=2,
            community_embedding_length=10,
            year_length=6,
            week_length=53,
            dropout_rate=0.1,
            estimate_uncertainty=True
        )
        
        # Test forward pass with dummy data
        batch_size = 4
        community = torch.randint(0, 10, (batch_size,))
        year = torch.randint(0, 6, (batch_size,))
        week = torch.randint(0, 53, (batch_size,))
        property_feat = torch.randn(batch_size, 3)
        time_feat = torch.randn(batch_size, 5)
        market_feat = torch.randn(batch_size, 2)
        
        output = model(community, year, week, property_feat, time_feat, market_feat)
        
        print(f"  ✓ Model instantiated successfully")
        print(f"  ✓ Forward pass works (output shape: {output.shape})")
        print("✓ Model architecture working\n")
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_directories():
    """Test that output directories can be created"""
    print("Testing directory structure...")
    
    directories = [
        'outputs',
        'outputs/models',
        'data/market_indicators',
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✓ {directory}")
        except Exception as e:
            print(f"  ✗ {directory} - ERROR: {e}")
            return False
    
    print("✓ Directory structure ready\n")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("INSTALLATION AND SETUP TEST")
    print("="*80)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("Custom Modules", test_module_imports),
        ("Market Indicators", test_market_indicators),
        ("Model Architecture", test_model_architecture),
        ("Directory Structure", test_directories),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("="*80)
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
        print("\n🎉 All tests passed! You're ready to train the model.")
        print("\nNext steps:")
        print("  1. Review QUICKSTART.md")
        print("  2. Run: python main_train_v2.py")
        print("  3. Test predictions: python predict_listings.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure data files are in the correct location")
        print("  - Check file permissions for output directories")
    
    print("="*80)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
