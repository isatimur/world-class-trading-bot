"""
Rate limiting utilities for the Trading Bot application.

This module provides rate limiting functionality for API calls,
user requests, and other operations that need throttling.
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple


class RateLimiter:
    """
    Rate limiter for controlling request frequency.
    
    This class implements a sliding window rate limiter that tracks
    requests within a specified time window and enforces rate limits.
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        cleanup_interval: int = 300
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
            cleanup_interval: Interval for cleaning up old entries (seconds)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        self.requests: Dict[str, list] = defaultdict(list)
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str) -> Tuple[bool, Optional[float]]:
        """
        Check if a request is allowed.
        
        Args:
            key: Unique identifier for the rate limit (e.g., user ID, API key)
            
        Returns:
            Tuple of (is_allowed, wait_time_seconds)
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup(current_time)
        
        # Get requests for this key
        requests = self.requests[key]
        
        # Remove requests outside the window
        window_start = current_time - self.window_seconds
        requests = [req_time for req_time in requests if req_time > window_start]
        self.requests[key] = requests
        
        # Check if under limit
        if len(requests) < self.max_requests:
            # Add current request
            requests.append(current_time)
            return True, None
        
        # Calculate wait time
        oldest_request = min(requests)
        wait_time = window_start - oldest_request
        
        return False, max(0, wait_time)
    
    async def wait_if_needed(self, key: str) -> bool:
        """
        Wait if rate limit is exceeded.
        
        Args:
            key: Unique identifier for the rate limit
            
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        is_allowed, wait_time = self.is_allowed(key)
        
        if not is_allowed and wait_time:
            await asyncio.sleep(wait_time)
            return self.is_allowed(key)[0]
        
        return is_allowed
    
    def _cleanup(self, current_time: float) -> None:
        """Clean up old entries from all keys."""
        window_start = current_time - self.window_seconds
        
        for key in list(self.requests.keys()):
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if req_time > window_start
            ]
            
            # Remove empty keys
            if not self.requests[key]:
                del self.requests[key]
        
        self.last_cleanup = current_time
    
    def get_stats(self, key: str) -> Dict[str, int]:
        """
        Get rate limit statistics for a key.
        
        Args:
            key: Unique identifier for the rate limit
            
        Returns:
            Dictionary with current request count and limit
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        requests = [
            req_time for req_time in self.requests[key] 
            if req_time > window_start
        ]
        
        return {
            "current_requests": len(requests),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "remaining_requests": max(0, self.max_requests - len(requests))
        }


class APIRateLimiter:
    """
    Specialized rate limiter for API endpoints.
    
    This class provides rate limiting for different API endpoints
    with configurable limits per endpoint.
    """
    
    def __init__(self):
        """Initialize API rate limiter with default limits."""
        self.limiters: Dict[str, RateLimiter] = {}
        self.default_limiter = RateLimiter(max_requests=100, window_seconds=60)
    
    def add_endpoint(
        self,
        endpoint: str,
        max_requests: int,
        window_seconds: int
    ) -> None:
        """
        Add rate limit for a specific endpoint.
        
        Args:
            endpoint: API endpoint path
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        """
        self.limiters[endpoint] = RateLimiter(max_requests, window_seconds)
    
    def is_allowed(self, endpoint: str, key: str) -> Tuple[bool, Optional[float]]:
        """
        Check if API request is allowed.
        
        Args:
            endpoint: API endpoint path
            key: User or API key identifier
            
        Returns:
            Tuple of (is_allowed, wait_time_seconds)
        """
        limiter = self.limiters.get(endpoint, self.default_limiter)
        return limiter.is_allowed(f"{endpoint}:{key}")
    
    async def wait_if_needed(self, endpoint: str, key: str) -> bool:
        """
        Wait if API rate limit is exceeded.
        
        Args:
            endpoint: API endpoint path
            key: User or API key identifier
            
        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        limiter = self.limiters.get(endpoint, self.default_limiter)
        return await limiter.wait_if_needed(f"{endpoint}:{key}")


# Global rate limiters
user_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
api_rate_limiter = APIRateLimiter() 