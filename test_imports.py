#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory
if not os.path.exists('logs'):
    os.makedirs('logs')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_imports.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports"""
    try:
        logger.info("Testing imports...")
        
        # Test config import
        try:
            from config import Config
            logger.info("✅ Config imported successfully")
        except ImportError as e:
            logger.error(f"❌ Config import failed: {e}")
            return False
        
        # Test module imports
        modules_to_test = [
            ('modules.data_collector', 'DataCollector'),
            ('modules.technical_analysis', 'TechnicalAnalysis'),
            ('modules.signal_manager', 'SignalManager'),
            ('modules.telegram_bot', 'TelegramBot'),
            ('modules.market_analysis', 'MarketAnalysis'),
            ('modules.performance', 'PerformanceAnalyzer'),
            ('modules.dynamic_strictness', 'DynamicStrictness'),
            ('modules.signal_tracker', 'SignalTracker'),
            ('modules.ai_model', 'AIModel'),
        ]
        
        for module_name, class_name in modules_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                class_obj = getattr(module, class_name)
                logger.info(f"✅ {module_name}.{class_name} imported successfully")
            except ImportError as e:
                logger.error(f"❌ {module_name}.{class_name} import failed: {e}")
                return False
            except AttributeError as e:
                logger.error(f"❌ {module_name}.{class_name} class not found: {e}")
                return False
        
        logger.info("✅ All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        logger.info("Testing basic functionality...")
        
        # Test config
        from config import Config
        Config.create_directories()
        logger.info("✅ Config directories created")
        
        # Test data collector
        from modules.data_collector import DataCollector
        data_collector = DataCollector()
        logger.info("✅ DataCollector initialized")
        
        # Test getting popular pairs
        try:
            popular_pairs = data_collector.get_popular_usdt_pairs(max_pairs=5)
            logger.info(f"✅ Got {len(popular_pairs)} popular pairs")
        except Exception as e:
            logger.warning(f"⚠️ Could not get popular pairs: {e}")
        
        logger.info("✅ Basic functionality test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting import and functionality tests...")
    
    if test_imports():
        logger.info("Import test passed!")
        if test_basic_functionality():
            logger.info("All tests passed! The system should work.")
        else:
            logger.error("Basic functionality test failed!")
    else:
        logger.error("Import test failed!") 