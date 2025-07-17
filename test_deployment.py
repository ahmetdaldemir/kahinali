#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports"""
    try:
        logger.info("Testing basic imports...")
        
        # Test basic Python modules
        import numpy as np
        import pandas as pd
        import requests
        
        logger.info("✅ Basic imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Basic import failed: {e}")
        return False

def test_config():
    """Test config import"""
    try:
        logger.info("Testing config import...")
        
        # Add current directory to Python path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from config import Config
        logger.info("✅ Config imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Config import failed: {e}")
        return False

def test_modules():
    """Test module imports"""
    try:
        logger.info("Testing module imports...")
        
        # Add current directory to Python path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Test module imports
        from modules.data_collector import DataCollector
        from modules.technical_analysis import TechnicalAnalysis
        from modules.signal_manager import SignalManager
        
        logger.info("✅ Module imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Module import failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting deployment test...")
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Config import", test_config),
        ("Module imports", test_modules),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} passed")
        else:
            logger.error(f"❌ {test_name} failed")
    
    logger.info(f"Test results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Deployment should work.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 