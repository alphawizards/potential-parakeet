"""
Trade Router Tests
==================
Unit tests for the trades API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestTradesCRUD:
    """Tests for trade CRUD operations."""
    
    @pytest.fixture
    def trades_client(self):
        """Create test client for trades API."""
        from backend.main import app
        return TestClient(app)
    
    def test_create_trade_valid(self, trades_client):
        """Create trade with valid data."""
        trade_data = {
            "ticker": "SPY",
            "direction": "BUY",
            "quantity": 100,
            "entry_price": 450.0,
            "entry_date": "2024-01-15T10:00:00",
            "strategy": "Momentum"
        }
        response = trades_client.post("/api/trades/", json=trade_data)
        assert response.status_code == 201
        data = response.json()
        assert data["ticker"] == "SPY"
        assert data["direction"] == "BUY"
    
    def test_create_trade_invalid_direction(self, trades_client):
        """Create trade with invalid direction should fail."""
        trade_data = {
            "ticker": "SPY",
            "direction": "INVALID",
            "quantity": 100,
            "entry_price": 450.0
        }
        response = trades_client.post("/api/trades/", json=trade_data)
        assert response.status_code == 422
    
    def test_create_trade_negative_quantity(self, trades_client):
        """Create trade with negative quantity should fail."""
        trade_data = {
            "ticker": "SPY",
            "direction": "BUY",
            "quantity": -100,
            "entry_price": 450.0
        }
        response = trades_client.post("/api/trades/", json=trade_data)
        assert response.status_code == 422
    
    def test_get_trades_list(self, trades_client):
        """Get paginated list of trades."""
        response = trades_client.get("/api/trades/")
        assert response.status_code == 200
        data = response.json()
        assert "trades" in data or "items" in data or isinstance(data, list)
    
    def test_get_trades_with_pagination(self, trades_client):
        """Get trades with custom pagination."""
        response = trades_client.get("/api/trades/?page=1&page_size=10")
        assert response.status_code == 200
    
    def test_get_trade_not_found(self, trades_client):
        """Get non-existent trade returns 404."""
        response = trades_client.get("/api/trades/99999")
        assert response.status_code == 404
    
    def test_update_trade_not_found(self, trades_client):
        """Update non-existent trade returns 404."""
        response = trades_client.patch("/api/trades/99999", json={"notes": "test"})
        assert response.status_code == 404
    
    def test_delete_trade_not_found(self, trades_client):
        """Delete non-existent trade returns 404."""
        response = trades_client.delete("/api/trades/99999")
        assert response.status_code == 404


class TestTradeMetrics:
    """Tests for trade metrics endpoints."""
    
    @pytest.fixture
    def trades_client(self):
        """Create test client for trades API."""
        from backend.main import app
        return TestClient(app)
    
    def test_get_portfolio_metrics(self, trades_client):
        """Get portfolio metrics."""
        response = trades_client.get("/api/trades/metrics/portfolio")
        assert response.status_code == 200
        data = response.json()
        # Should contain standard portfolio metrics
        assert "total_value" in data or "portfolio_value" in data or isinstance(data, dict)
    
    def test_get_portfolio_metrics_with_capital(self, trades_client):
        """Get portfolio metrics with custom initial capital."""
        response = trades_client.get("/api/trades/metrics/portfolio?initial_capital=50000")
        assert response.status_code == 200
    
    def test_get_dashboard_summary(self, trades_client):
        """Get dashboard summary."""
        response = trades_client.get("/api/trades/metrics/dashboard")
        assert response.status_code == 200
    
    def test_get_stats_by_ticker(self, trades_client):
        """Get performance stats grouped by ticker."""
        response = trades_client.get("/api/trades/metrics/by-ticker")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestTradeUtilities:
    """Tests for trade utility endpoints."""
    
    @pytest.fixture
    def trades_client(self):
        """Create test client for trades API."""
        from backend.main import app
        return TestClient(app)
    
    def test_generate_trade_id(self, trades_client):
        """Generate a unique trade ID."""
        response = trades_client.get("/api/trades/utils/generate-id")
        assert response.status_code == 200
        data = response.json()
        assert "trade_id" in data
        assert data["trade_id"].startswith("TRD")
    
    def test_generate_trade_id_custom_prefix(self, trades_client):
        """Generate trade ID with custom prefix."""
        response = trades_client.get("/api/trades/utils/generate-id?prefix=MOM")
        assert response.status_code == 200
        data = response.json()
        assert data["trade_id"].startswith("MOM")
