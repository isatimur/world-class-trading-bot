"""
Pytest configuration for trading bot tests.

This file handles import paths and common test fixtures.
"""

import sys
from pathlib import Path
import pytest
import asyncio
from unittest.mock import MagicMock

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure asyncio for pytest
pytest_plugins = ["pytest_asyncio"]

# Common fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from trading_bot.config.settings import Settings
    
    # Create a mock settings object
    mock_settings = MagicMock()
    mock_settings.GOOGLE_API_KEY = "test_google_api_key"
    mock_settings.BYBIT_API_KEY = "test_bybit_api_key"
    mock_settings.BYBIT_API_SECRET = "test_bybit_secret"
    mock_settings.TELEGRAM_BOT_TOKEN = "test_telegram_token"
    mock_settings.TRADING_MODE = "PAPER"
    mock_settings.RISK_TOLERANCE = "MODERATE"
    
    return mock_settings

@pytest.fixture
def mock_bybit_response():
    """Mock Bybit API response."""
    return {
        "success": True,
        "data": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "120000.00",
                    "price24hPcnt": "0.05",
                    "volume24h": "1000000"
                }
            ]
        }
    }

@pytest.fixture
def mock_google_ai_response():
    """Mock Google AI response."""
    return {
        "summary": "Test trading signal summary",
        "action_recommendation": "Test action recommendation",
        "risk_assessment": "Test risk assessment",
        "confidence_level": 0.85,
        "reasoning": "Test reasoning"
    } 