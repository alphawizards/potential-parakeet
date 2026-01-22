/**
 * useTruthEngine Hook
 * ===================
 * Custom hook for fetching Truth Engine strategy validation data.
 * Implements API fetch with fallback to mock data.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { StrategyMetrics } from '../types/strategy';
import { MOCK_STRATEGIES, GRAVEYARD_STATS } from '../components/truth-engine/mockData';

export interface GraveyardStats {
    total_trials_tested: number;
    trials_accepted: number;
    trials_rejected: number;
    acceptance_rate: number;
}

export interface TruthEngineData {
    strategies: StrategyMetrics[];
    graveyardStats: GraveyardStats;
    loading: boolean;
    error: string | null;
    refetch: () => void;
}

/**
 * Fetch Truth Engine strategy validation data.
 * Falls back to mock data if API is unavailable.
 * 
 * @param universe - Stock universe code (SPX500, ASX200, etc.)
 * @returns TruthEngineData with strategies, graveyard stats, loading state
 */
export function useTruthEngine(universe: string = 'SPX500'): TruthEngineData {
    const [strategies, setStrategies] = useState<StrategyMetrics[]>(MOCK_STRATEGIES);
    const [graveyardStats, setGraveyardStats] = useState<GraveyardStats>(GRAVEYARD_STATS);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/quant2/truth-engine/strategies?universe=${universe}`);

            if (!response.ok) {
                throw new Error(`API returned ${response.status}`);
            }

            const data = await response.json();
            console.log('Loaded Truth Engine data from API:', data);

            // Transform API response to match frontend types
            if (data.strategies && Array.isArray(data.strategies)) {
                setStrategies(data.strategies);
            }

            if (data.graveyard_stats) {
                setGraveyardStats({
                    total_trials_tested: data.graveyard_stats.total_trials_tested,
                    trials_accepted: data.graveyard_stats.trials_accepted,
                    trials_rejected: data.graveyard_stats.trials_rejected,
                    acceptance_rate: data.graveyard_stats.acceptance_rate
                });
            }
        } catch (err) {
            console.warn('API not available, using mock data:', err);
            // Keep mock data as fallback (already set as default)
            setStrategies(MOCK_STRATEGIES);
            setGraveyardStats(GRAVEYARD_STATS);
            setError(err instanceof Error ? err.message : 'Failed to load data');
        } finally {
            setLoading(false);
        }
    }, [universe]);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    const refetch = useCallback(() => {
        fetchData();
    }, [fetchData]);

    return {
        strategies,
        graveyardStats,
        loading,
        error,
        refetch
    };
}

// Summary stats derived from strategies
export interface SummaryStats {
    total: number;
    significant: number;
    rejected: number;
    avgDSR: string;
    avgPSR: string;
    totalTrials: number;
}

/**
 * Calculate summary statistics from strategies.
 * 
 * @param strategies - List of strategy metrics
 * @param graveyardStats - Graveyard statistics
 * @returns Summary stats for dashboard display
 */
export function useSummaryStats(
    strategies: StrategyMetrics[],
    graveyardStats: GraveyardStats
): SummaryStats {
    return useMemo(() => {
        if (!strategies.length) {
            return {
                total: 0,
                significant: 0,
                rejected: 0,
                avgDSR: '0.00',
                avgPSR: '0.0',
                totalTrials: 0
            };
        }

        const significant = strategies.filter(s => s.validity.is_significant);
        const avgDSR = strategies.reduce((sum, s) => sum + s.validity.dsr, 0) / strategies.length;
        const avgPSR = strategies.reduce((sum, s) => sum + s.validity.psr, 0) / strategies.length;

        return {
            total: strategies.length,
            significant: significant.length,
            rejected: strategies.length - significant.length,
            avgDSR: avgDSR.toFixed(2),
            avgPSR: (avgPSR * 100).toFixed(1),
            totalTrials: graveyardStats.total_trials_tested
        };
    }, [strategies, graveyardStats]);
}

export default useTruthEngine;
