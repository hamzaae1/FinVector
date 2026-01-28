"""
FinVector - Enhanced Logging Module
Provides structured logging for search queries, API calls, and performance metrics
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
from functools import wraps
import time

# Create logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler for persistent logs
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, 'finvector.log'),
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
))

# Search-specific logger
search_logger = logging.getLogger('finvector.search')
search_logger.addHandler(file_handler)

# API logger
api_logger = logging.getLogger('finvector.api')
api_logger.addHandler(file_handler)

# Performance logger
perf_logger = logging.getLogger('finvector.performance')
perf_logger.addHandler(file_handler)


class SearchLogger:
    """Structured logging for search operations"""

    @staticmethod
    def log_search(
        query: str,
        max_budget: float,
        results_count: int,
        response_time_ms: float,
        search_type: str = "text",
        filters: Optional[Dict] = None
    ):
        """Log a search query"""
        log_entry = {
            "event": "search",
            "search_type": search_type,
            "query": query,
            "max_budget": max_budget,
            "results_count": results_count,
            "response_time_ms": round(response_time_ms, 2),
            "filters": filters or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        search_logger.info(json.dumps(log_entry))

    @staticmethod
    def log_alternatives(
        product_id: str,
        max_budget: float,
        alternatives_count: int,
        response_time_ms: float
    ):
        """Log alternatives search"""
        log_entry = {
            "event": "alternatives",
            "product_id": product_id,
            "max_budget": max_budget,
            "alternatives_count": alternatives_count,
            "response_time_ms": round(response_time_ms, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        search_logger.info(json.dumps(log_entry))

    @staticmethod
    def log_budget_stretcher(
        product_id: str,
        total_budget: float,
        success: bool,
        bundle_total: Optional[float] = None,
        savings: Optional[float] = None,
        response_time_ms: float = 0
    ):
        """Log budget stretcher request"""
        log_entry = {
            "event": "budget_stretcher",
            "product_id": product_id,
            "total_budget": total_budget,
            "success": success,
            "bundle_total": bundle_total,
            "savings": savings,
            "response_time_ms": round(response_time_ms, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        search_logger.info(json.dumps(log_entry))


class APILogger:
    """Structured logging for API requests"""

    @staticmethod
    def log_request(
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        client_ip: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log an API request"""
        log_entry = {
            "event": "api_request",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
            "client_ip": client_ip,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        api_logger.info(json.dumps(log_entry))


class PerformanceLogger:
    """Performance metrics logging"""

    @staticmethod
    def log_metric(
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ):
        """Log a performance metric"""
        log_entry = {
            "event": "performance",
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        perf_logger.info(json.dumps(log_entry))


def timed_operation(operation_name: str):
    """Decorator to time and log function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                PerformanceLogger.log_metric(
                    operation=operation_name,
                    duration_ms=duration_ms,
                    metadata={"status": "success"}
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                PerformanceLogger.log_metric(
                    operation=operation_name,
                    duration_ms=duration_ms,
                    metadata={"status": "error", "error": str(e)}
                )
                raise
        return wrapper
    return decorator


# Analytics helper
class SearchAnalytics:
    """Helper to analyze search logs"""

    @staticmethod
    def get_recent_searches(limit: int = 100) -> list:
        """Get recent search queries from log file"""
        log_file = os.path.join(LOG_DIR, 'finvector.log')
        if not os.path.exists(log_file):
            return []

        searches = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    if '"event": "search"' in line:
                        # Extract JSON from log line
                        json_start = line.find('{')
                        if json_start != -1:
                            entry = json.loads(line[json_start:])
                            searches.append(entry)
                except:
                    continue

        return searches[-limit:]

    @staticmethod
    def get_average_response_time() -> float:
        """Calculate average response time from logs"""
        searches = SearchAnalytics.get_recent_searches()
        if not searches:
            return 0.0
        times = [s.get('response_time_ms', 0) for s in searches]
        return sum(times) / len(times)

    @staticmethod
    def get_popular_queries(limit: int = 10) -> list:
        """Get most popular search queries"""
        searches = SearchAnalytics.get_recent_searches(1000)
        query_counts = {}
        for s in searches:
            query = s.get('query', '')
            query_counts[query] = query_counts.get(query, 0) + 1

        sorted_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_queries[:limit]
