"""
Quant 2.0 API Endpoint Tests
============================
Tests for the new Quant 2.0 endpoints:
- /api/quant2/regime
- /api/quant2/stat-arb
- /api/quant2/meta-labeling
"""

import pytest
from fastapi.testclient import TestClient


class TestQuant2Endpoints:
    """Test suite for Quant 2.0 API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        from backend.main import app
        return TestClient(app)

    # ============== Regime Detection Endpoint ==============

    def test_regime_endpoint_returns_200(self, client):
        """Test that regime endpoint returns 200 OK."""
        response = client.get("/api/quant2/regime")
        assert response.status_code == 200

    def test_regime_endpoint_with_universe(self, client):
        """Test regime endpoint with universe parameter."""
        response = client.get("/api/quant2/regime?universe=SPX500")
        assert response.status_code == 200
        data = response.json()
        assert data["universe"] == "SPX500"

    def test_regime_response_structure(self, client):
        """Test that regime response has expected structure."""
        response = client.get("/api/quant2/regime")
        data = response.json()
        
        # Check required fields
        assert "universe" in data
        assert "generated_at" in data
        assert "current_regime" in data
        assert "regime_probabilities" in data
        assert "days_in_regime" in data
        assert "vix_level" in data
        assert "regime_history" in data

    def test_regime_probabilities_valid(self, client):
        """Test that regime probabilities are valid (0-1 range)."""
        response = client.get("/api/quant2/regime")
        data = response.json()
        probs = data["regime_probabilities"]
        
        assert "bull" in probs
        assert "bear" in probs
        assert "chop" in probs
        assert 0 <= probs["bull"] <= 1
        assert 0 <= probs["bear"] <= 1
        assert 0 <= probs["chop"] <= 1

    def test_current_regime_valid(self, client):
        """Test that current regime is one of expected values."""
        response = client.get("/api/quant2/regime")
        data = response.json()
        assert data["current_regime"] in ["BULL", "BEAR", "CHOP"]

    # ============== Statistical Arbitrage Endpoint ==============

    def test_stat_arb_endpoint_returns_200(self, client):
        """Test that stat-arb endpoint returns 200 OK."""
        response = client.get("/api/quant2/stat-arb")
        assert response.status_code == 200

    def test_stat_arb_response_structure(self, client):
        """Test that stat-arb response has expected structure."""
        response = client.get("/api/quant2/stat-arb")
        data = response.json()
        
        # Check required fields
        assert "universe" in data
        assert "generated_at" in data
        assert "total_pairs" in data
        assert "active_signals" in data
        assert "pairs" in data
        assert isinstance(data["pairs"], list)

    def test_stat_arb_pair_structure(self, client):
        """Test that each pair has expected fields."""
        response = client.get("/api/quant2/stat-arb")
        data = response.json()
        
        if len(data["pairs"]) > 0:
            pair = data["pairs"][0]
            assert "pair_id" in pair
            assert "stock_a" in pair
            assert "stock_b" in pair
            assert "z_score" in pair
            assert "half_life" in pair
            assert "correlation" in pair
            assert "signal" in pair

    def test_stat_arb_signals_valid(self, client):
        """Test that pair signals are valid values."""
        response = client.get("/api/quant2/stat-arb")
        data = response.json()
        
        valid_signals = ["LONG_SPREAD", "SHORT_SPREAD", "NEUTRAL"]
        for pair in data["pairs"]:
            assert pair["signal"] in valid_signals

    # ============== Meta-Labeling Endpoint ==============

    def test_meta_labeling_endpoint_returns_200(self, client):
        """Test that meta-labeling endpoint returns 200 OK."""
        response = client.get("/api/quant2/meta-labeling")
        assert response.status_code == 200

    def test_meta_labeling_response_structure(self, client):
        """Test that meta-labeling response has expected structure."""
        response = client.get("/api/quant2/meta-labeling")
        data = response.json()
        
        # Check required fields
        assert "universe" in data
        assert "generated_at" in data
        assert "model_auc" in data
        assert "total_signals" in data
        assert "accepted_signals" in data
        assert "signals" in data
        assert isinstance(data["signals"], list)

    def test_meta_labeling_signal_structure(self, client):
        """Test that each signal has expected fields."""
        response = client.get("/api/quant2/meta-labeling")
        data = response.json()
        
        if len(data["signals"]) > 0:
            signal = data["signals"][0]
            assert "ticker" in signal
            assert "base_signal" in signal
            assert "meta_label" in signal
            assert "confidence" in signal
            assert "position_size" in signal
            assert "features" in signal

    def test_meta_labeling_labels_valid(self, client):
        """Test that meta labels are valid values."""
        response = client.get("/api/quant2/meta-labeling")
        data = response.json()
        
        for signal in data["signals"]:
            assert signal["base_signal"] in ["BUY", "SELL"]
            assert signal["meta_label"] in ["ACCEPT", "REJECT"]
            assert 0 <= signal["confidence"] <= 1

    def test_meta_labeling_rejection_rate(self, client):
        """Test that rejection rate is calculated correctly."""
        response = client.get("/api/quant2/meta-labeling")
        data = response.json()
        
        if data["total_signals"] > 0:
            expected_rate = 1 - (data["accepted_signals"] / data["total_signals"])
            assert abs(data["rejection_rate"] - expected_rate) < 0.01


class TestQuant2EndpointEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    def test_invalid_universe_handled(self, client):
        """Test that invalid universe parameter is handled gracefully."""
        response = client.get("/api/quant2/regime?universe=INVALID")
        # Should return 200 (falls back to default) or 400/422 (validation error)
        assert response.status_code in [200, 400, 422]

    def test_empty_pairs_handled(self, client):
        """Test that empty pairs list is handled gracefully."""
        response = client.get("/api/quant2/stat-arb")
        data = response.json()
        # Should return valid counts even if no active signals
        assert data["total_pairs"] >= 0
        assert data["active_signals"] >= 0


class TestTruthEngineEndpoint:
    """Test suite for Truth Engine API endpoint."""

    @pytest.fixture
    def client(self):
        from backend.main import app
        return TestClient(app)

    def test_truth_engine_endpoint_returns_200(self, client):
        """Test that truth engine endpoint returns 200 OK."""
        response = client.get("/api/quant2/truth-engine/strategies")
        assert response.status_code == 200

    def test_truth_engine_response_structure(self, client):
        """Test that truth engine response has expected structure."""
        response = client.get("/api/quant2/truth-engine/strategies")
        data = response.json()
        
        # Check required fields
        assert "universe" in data
        assert "generated_at" in data
        assert "strategies" in data
        assert "graveyard_stats" in data
        assert isinstance(data["strategies"], list)

    def test_truth_engine_strategies_not_empty(self, client):
        """Test that strategies list is not empty."""
        response = client.get("/api/quant2/truth-engine/strategies")
        data = response.json()
        assert len(data["strategies"]) > 0

    def test_truth_engine_strategy_structure(self, client):
        """Test that each strategy has expected fields."""
        response = client.get("/api/quant2/truth-engine/strategies")
        data = response.json()
        
        if len(data["strategies"]) > 0:
            strategy = data["strategies"][0]
            assert "id" in strategy
            assert "name" in strategy
            assert "returns" in strategy
            assert "risk" in strategy
            assert "efficiency" in strategy
            assert "validity" in strategy
            assert "regime_performance" in strategy
            assert "equity_curve" in strategy
            assert "drawdown_series" in strategy

    def test_truth_engine_validity_metrics(self, client):
        """Test that validity metrics are present."""
        response = client.get("/api/quant2/truth-engine/strategies")
        data = response.json()
        
        if len(data["strategies"]) > 0:
            validity = data["strategies"][0]["validity"]
            assert "psr" in validity
            assert "dsr" in validity
            assert "num_trials" in validity
            assert "is_significant" in validity
            assert 0 <= validity["psr"] <= 1

    def test_truth_engine_graveyard_stats(self, client):
        """Test that graveyard stats are valid."""
        response = client.get("/api/quant2/truth-engine/strategies")
        data = response.json()
        
        stats = data["graveyard_stats"]
        assert stats["total_trials_tested"] >= 0
        assert stats["trials_accepted"] >= 0
        assert stats["trials_rejected"] >= 0
        assert 0 <= stats["acceptance_rate"] <= 1
