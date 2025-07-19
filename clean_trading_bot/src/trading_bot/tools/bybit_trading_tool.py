"""
Bybit V5 API Trading Tool
Integration with Bybit exchange for cryptocurrency trading
"""
import asyncio
import hashlib
import hmac
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlencode
import aiohttp
import pandas as pd

from ..config.settings import Settings

class BybitTradingTool:
    """Bybit V5 API trading tool for cryptocurrency trading"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        """Initialize Bybit trading tool"""
        self.api_key = api_key or Settings.BYBIT_API_KEY
        self.api_secret = api_secret or Settings.BYBIT_API_SECRET
        self.testnet = testnet if testnet is not None else Settings.BYBIT_TESTNET
        
        # API endpoints
        if self.testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit = 120  # requests per minute
        
        # Session for HTTP requests
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, recv_window: str, params: str) -> str:
        """Generate HMAC signature for API authentication"""
        param_str = timestamp + self.api_key + recv_window + params
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request to Bybit API"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < 60/self.rate_limit:
            await asyncio.sleep(60/self.rate_limit - (current_time - self.last_request_time))
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'TradingAgent/1.0'
        }
        
        if signed:
            timestamp = str(int(time.time() * 1000))
            recv_window = "5000"
            
            if params:
                param_str = urlencode(params)
            else:
                param_str = ""
            
            signature = self._generate_signature(timestamp, recv_window, param_str)
            
            headers.update({
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-SIGN-TYPE': '2',
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-RECV-WINDOW': recv_window
            })
        
        try:
            if method.upper() == 'GET':
                if params:
                    url += '?' + urlencode(params)
                async with self.session.get(url, headers=headers) as response:
                    data = await response.json()
            else:
                async with self.session.post(url, headers=headers, json=params) as response:
                    data = await response.json()
            
            self.request_count += 1
            self.last_request_time = time.time()
            
            if data.get('retCode') == 0:
                return {
                    'success': True,
                    'data': data.get('result', {}),
                    'response': data
                }
            else:
                return {
                    'success': False,
                    'error': data.get('retMsg', 'Unknown error'),
                    'code': data.get('retCode'),
                    'response': data
                }
                
        except Exception as e:
            self.logger.error(f"Bybit API request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        endpoint = "/v5/account/wallet-balance"
        params = {
            'accountType': 'UNIFIED'
        }
        return await self._make_request('GET', endpoint, params, signed=True)
    
    async def get_positions(self, category: str = 'linear', symbol: str = None) -> Dict[str, Any]:
        """Get current positions"""
        endpoint = "/v5/position/list"
        params = {
            'category': category
        }
        if symbol:
            params['symbol'] = symbol
        return await self._make_request('GET', endpoint, params, signed=True)
    
    async def get_market_data(self, symbol: str, interval: str = '1', limit: int = 200) -> Dict[str, Any]:
        """Get market data (candlesticks)"""
        endpoint = "/v5/market/kline"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return await self._make_request('GET', endpoint, params)
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information"""
        endpoint = "/v5/market/tickers"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        return await self._make_request('GET', endpoint, params)
    
    async def get_orderbook(self, symbol: str, limit: int = 25) -> Dict[str, Any]:
        """Get order book"""
        endpoint = "/v5/market/orderbook"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': limit
        }
        return await self._make_request('GET', endpoint, params)
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         qty: float, price: float = None, 
                         stop_loss: float = None, take_profit: float = None,
                         time_in_force: str = 'GTC') -> Dict[str, Any]:
        """Place a new order"""
        endpoint = "/v5/order/create"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side.upper(),
            'orderType': order_type.upper(),
            'qty': str(qty),
            'timeInForce': time_in_force
        }
        
        if price and order_type.upper() == 'LIMIT':
            params['price'] = str(price)
        
        if stop_loss:
            params['stopLoss'] = str(stop_loss)
        
        if take_profit:
            params['takeProfit'] = str(take_profit)
        
        return await self._make_request('POST', endpoint, params, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: str = None, 
                          order_link_id: str = None) -> Dict[str, Any]:
        """Cancel an order"""
        endpoint = "/v5/order/cancel"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif order_link_id:
            params['orderLinkId'] = order_link_id
        else:
            return {
                'success': False,
                'error': 'Either orderId or orderLinkId must be provided'
            }
        
        return await self._make_request('POST', endpoint, params, signed=True)
    
    async def get_open_orders(self, symbol: str = None) -> Dict[str, Any]:
        """Get open orders"""
        endpoint = "/v5/order/realtime"
        params = {
            'category': 'linear'
        }
        if symbol:
            params['symbol'] = symbol
        return await self._make_request('GET', endpoint, params, signed=True)
    
    async def get_order_history(self, symbol: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get order history"""
        endpoint = "/v5/order/history"
        params = {
            'category': 'linear',
            'limit': limit
        }
        if symbol:
            params['symbol'] = symbol
        return await self._make_request('GET', endpoint, params, signed=True)
    
    async def get_trade_history(self, symbol: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get trade history"""
        endpoint = "/v5/execution/list"
        params = {
            'category': 'linear',
            'limit': limit
        }
        if symbol:
            params['symbol'] = symbol
        return await self._make_request('GET', endpoint, params, signed=True)
    
    async def get_available_symbols(self, category: str = 'linear') -> Dict[str, Any]:
        """Get available trading symbols"""
        endpoint = "/v5/market/instruments-info"
        params = {
            'category': category
        }
        return await self._make_request('GET', endpoint, params)
    
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get funding rate for perpetual contracts"""
        endpoint = "/v5/market/funding/history"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': 1
        }
        return await self._make_request('GET', endpoint, params)
    
    async def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour ticker statistics"""
        endpoint = "/v5/market/tickers"
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        return await self._make_request('GET', endpoint, params)
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview for all symbols"""
        endpoint = "/v5/market/tickers"
        params = {
            'category': 'linear'
        }
        return await self._make_request('GET', endpoint, params)
    
    async def analyze_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Analyze position risk for a symbol"""
        try:
            # Get current position
            position_result = await self.get_positions(symbol=symbol)
            if not position_result['success']:
                return position_result
            
            # Get market data
            market_result = await self.get_market_data(symbol, interval='1', limit=100)
            if not market_result['success']:
                return market_result
            
            # Get ticker for current price
            ticker_result = await self.get_ticker(symbol)
            if not ticker_result['success']:
                return ticker_result
            
            positions = position_result['data'].get('list', [])
            market_data = market_result['data'].get('list', [])
            ticker_data = ticker_result['data'].get('list', [])
            
            if not positions or not market_data or not ticker_data:
                return {
                    'success': False,
                    'error': 'No data available for analysis'
                }
            
            position = positions[0]
            current_price = float(ticker_data[0]['lastPrice'])
            
            # Calculate risk metrics
            position_size = float(position.get('size', 0))
            entry_price = float(position.get('avgPrice', 0))
            unrealized_pnl = float(position.get('unrealisedPnl', 0))
            
            if position_size > 0 and entry_price > 0:
                # Calculate percentage change
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Calculate position value
                position_value = position_size * current_price
                
                # Risk assessment
                risk_level = 'LOW'
                if abs(price_change_pct) > 10:
                    risk_level = 'HIGH'
                elif abs(price_change_pct) > 5:
                    risk_level = 'MODERATE'
                
                return {
                    'success': True,
                    'data': {
                        'symbol': symbol,
                        'position_size': position_size,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'price_change_pct': price_change_pct,
                        'position_value': position_value,
                        'risk_level': risk_level,
                        'side': position.get('side', 'NONE')
                    }
                }
            else:
                return {
                    'success': True,
                    'data': {
                        'symbol': symbol,
                        'position_size': 0,
                        'message': 'No active position'
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Position risk analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            # Get account balance
            balance_result = await self.get_account_info()
            if not balance_result['success']:
                return balance_result
            
            # Get all positions
            positions_result = await self.get_positions()
            if not positions_result['success']:
                return positions_result
            
            balance_data = balance_result['data'].get('list', [])
            positions_data = positions_result['data'].get('list', [])
            
            if not balance_data:
                return {
                    'success': False,
                    'error': 'No account data available'
                }
            
            account = balance_data[0]
            total_balance = float(account.get('totalWalletBalance', 0))
            total_unrealized_pnl = float(account.get('totalUnrealisedPnl', 0))
            total_realized_pnl = float(account.get('totalRealisedPnl', 0))
            
            # Calculate portfolio metrics
            active_positions = [p for p in positions_data if float(p.get('size', 0)) > 0]
            total_positions = len(active_positions)
            
            # Calculate total position value
            total_position_value = sum(float(p.get('positionValue', 0)) for p in active_positions)
            
            # Calculate portfolio allocation
            if total_balance > 0:
                position_allocation = (total_position_value / total_balance) * 100
            else:
                position_allocation = 0
            
            return {
                'success': True,
                'data': {
                    'total_balance': total_balance,
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'total_realized_pnl': total_realized_pnl,
                    'total_positions': total_positions,
                    'total_position_value': total_position_value,
                    'position_allocation_pct': position_allocation,
                    'active_positions': active_positions,
                    'account_type': account.get('accountType', 'UNKNOWN')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio summary failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Convenience functions for easy integration
async def get_bybit_market_data(symbol: str, interval: str, limit: int) -> Dict[str, Any]:
    """Get Bybit market data"""
    async with BybitTradingTool() as bybit:
        return await bybit.get_market_data(symbol, interval, limit)

async def get_bybit_account_info() -> Dict[str, Any]:
    """Get Bybit account information"""
    async with BybitTradingTool() as bybit:
        return await bybit.get_account_info()

async def get_bybit_positions(symbol: str = None) -> Dict[str, Any]:
    """Get Bybit positions"""
    async with BybitTradingTool() as bybit:
        return await bybit.get_positions(symbol=symbol)

async def place_bybit_order(symbol: str, side: str, order_type: str, 
                           qty: float, price: float = None) -> Dict[str, Any]:
    """Place Bybit order"""
    async with BybitTradingTool() as bybit:
        return await bybit.place_order(symbol, side, order_type, qty, price)

async def get_bybit_portfolio_summary() -> Dict[str, Any]:
    """Get Bybit portfolio summary"""
    async with BybitTradingTool() as bybit:
        return await bybit.get_portfolio_summary() 